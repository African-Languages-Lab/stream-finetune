"""Push the exact checkpoints that produced the verified-good audio to HF, one repo per language.

Unlike build_and_push_proven_hf_bundles.py (which re-resolves "best recent" checkpoints from
tensorboard), this script is MANIFEST-DRIVEN: it uploads precisely the llm/flow checkpoint pair
that generate_best_all_languages.py recorded for each language, so the hosted model reproduces
the audio that was actually listened to and approved -- including where that pair came from the
archived run.

Shared assets (cosyvoice3.yaml, campplus.onnx, speech_tokenizer_v3.onnx, the original pretrained
hift vocoder, and the CosyVoice-BlankEN tokenizer) are copied from the pretrained release, which
is the same state the manifest's generation run used, so the tokenizer matches what produced the
approved audio.

Each bundle is deleted from local scratch immediately after a successful upload -- 18 languages
at ~5GB each will not sit on disk at once. Re-running skips languages already complete on the Hub.
"""
import argparse
import json
import os
import shutil

import torch
from huggingface_hub import HfApi

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
BUNDLE_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/hf_bundles_good_audio"
ORG = "all-lab"
COLLECTION_TITLE = "CosyVoice3 TTS — African Language Fine-tunes"

SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
REQUIRED = ["llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v3.onnx", "cosyvoice3.yaml"]

# Verified clean in the best-checkpoint audit: full-length output, no HF artefacts,
# healthy level. Excludes collapsed (af-ZA, en-UG, ki-KE, nd-ZW, rw-RW, yo-NG),
# noisy (sn-ZW), too-quiet (wo-SN) and failed (bem-ZM) languages.
GOOD = ["am-ET", "ar-AR", "ber-MA", "ee-GH", "ff-SN", "ha-NG", "ig-NG", "lg-UG", "ln-CD",
        "mg-MG", "nso-ZA", "ny-MW", "or-KE", "so-SO", "st-ZA", "sw-KE", "ti-ER", "tn-BW",
        "ts-ZA", "tw-GH", "umb-AO", "ve-ZA", "xh-ZA", "zu-ZA"]
ALREADY_ON_HF = ["ber-MA", "ee-GH", "ha-NG", "ig-NG", "tw-GH", "umb-AO"]


def clean(src, dst):
    """Strip optimizer/scheduler state -- inference only needs the tensors."""
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def already_complete(api, repo_id):
    try:
        files = set(api.list_repo_files(repo_id))
    except Exception:
        return False
    return all(f in files for f in REQUIRED) and any(f.startswith("CosyVoice-BlankEN/") for f in files)


def build_bundle(lang, entry):
    bundle_dir = os.path.join(BUNDLE_ROOT, lang)
    os.makedirs(bundle_dir, exist_ok=True)

    llm_ckpt, flow_ckpt = entry["llm_checkpoint"], entry["flow_checkpoint"]
    for p in (llm_ckpt, flow_ckpt):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{lang}: missing checkpoint {p}")

    print(f"[{lang}] llm  {entry['llm_source']:8} {llm_ckpt}", flush=True)
    print(f"[{lang}] flow {entry['flow_source']:8} {flow_ckpt}", flush=True)

    clean(llm_ckpt, os.path.join(bundle_dir, "llm.pt"))
    clean(flow_ckpt, os.path.join(bundle_dir, "flow.pt"))

    for asset in SHARED:
        shutil.copy2(os.path.join(PRETRAINED_DIR, asset), os.path.join(bundle_dir, asset))
    dst = os.path.join(bundle_dir, "CosyVoice-BlankEN")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(PRETRAINED_DIR, "CosyVoice-BlankEN"), dst)

    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(bundle_dir) for f in fs) / 2 ** 30
    print(f"[{lang}] bundle built, {size:.2f} GB", flush=True)
    return bundle_dir


def write_card(bundle_dir, lang, entry):
    q = entry.get("result", {})
    card = f"""---
license: apache-2.0
language: {lang.split('-')[0]}
pipeline_tag: text-to-speech
tags: [cosyvoice, cosyvoice3, tts, african-languages, zero-shot]
---

# CosyVoice3 — {lang}

Fine-tuned CosyVoice3-0.5B for **{lang}**, African Languages Lab.

This is the exact llm + flow checkpoint pair that produced the verified-good sample in the
best-checkpoint audit ({q.get('duration', '?')}s output, peak {round(q.get('peak', 0), 3)}),
bundled with the original pretrained HiFT vocoder.

| component | source |
|---|---|
| llm | `{entry['llm_source']}` |
| flow | `{entry['flow_source']}` |
| vocoder | pretrained (not fine-tuned) |

## Usage

```python
from cosyvoice.cli.cosyvoice import CosyVoice3

model = CosyVoice3("{ORG}/cosyvoice3-individual-{lang}", fp16=False)
out = list(model.inference_zero_shot(
    "text to speak",
    "<transcript of reference clip><|endofprompt|>",
    "reference.wav",
    stream=False))
audio = out[0]["tts_speech"]
```

Zero-shot cloning: it speaks in the voice of whatever reference clip you supply, and the
reference transcript must end with `<|endofprompt|>`.
"""
    with open(os.path.join(bundle_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--langs", nargs="+", default=None)
    ap.add_argument("--include-existing", action="store_true",
                    help="also rebuild languages already published")
    ap.add_argument("--keep-bundles", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    langs = args.langs or [l for l in GOOD if args.include_existing or l not in ALREADY_ON_HF]

    token = os.environ.get("HF_TOKEN_ORGOWNER") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    print("authenticated as:", api.whoami().get("name"), flush=True)

    coll = next((c for c in api.list_collections(owner=ORG) if c.title == COLLECTION_TITLE), None)
    if coll is None:
        print(f"WARNING: collection '{COLLECTION_TITLE}' not found; skipping collection add", flush=True)

    os.makedirs(BUNDLE_ROOT, exist_ok=True)
    done, failed, skipped = [], [], []

    for i, lang in enumerate(langs, 1):
        repo_id = f"{ORG}/cosyvoice3-individual-{lang}"
        print(f"\n===== [{i}/{len(langs)}] {lang} -> {repo_id} =====", flush=True)
        if already_complete(api, repo_id):
            print(f"[{lang}] already complete on the Hub, skipping", flush=True)
            skipped.append(lang)
            continue
        try:
            entry = manifest[lang]
            bundle_dir = build_bundle(lang, entry)
            write_card(bundle_dir, lang, entry)

            api.create_repo(repo_id=repo_id, exist_ok=True, private=True, repo_type="model")
            print(f"[{lang}] uploading...", flush=True)
            api.upload_folder(repo_id=repo_id, folder_path=bundle_dir, repo_type="model")
            if coll is not None:
                try:
                    api.add_collection_item(coll.slug, item_id=repo_id,
                                            item_type="model", exists_ok=True)
                except Exception as e:
                    print(f"[{lang}] collection add failed (non-fatal): {e}", flush=True)
            print(f"[{lang}] DONE https://huggingface.co/{repo_id}", flush=True)
            done.append(lang)
            if not args.keep_bundles:
                shutil.rmtree(bundle_dir, ignore_errors=True)
        except Exception as e:
            print(f"[{lang}] FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append((lang, str(e)))

    print("\n================ SUMMARY ================", flush=True)
    print(f"pushed  ({len(done)}): {done}", flush=True)
    print(f"skipped ({len(skipped)}): {skipped}", flush=True)
    print(f"failed  ({len(failed)}): {[l for l, _ in failed]}", flush=True)
    for l, e in failed:
        print(f"   {l}: {e[:200]}", flush=True)


if __name__ == "__main__":
    main()
