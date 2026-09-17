"""
Pushes Hausa, Twi, and Igbo as ONE self-contained CosyVoice3 model repo each -- best RECENT
(current run only, never archived) llm + best RECENT flow + the original pretrained vocoder,
matching the official release's own directory layout (loadable directly via
CosyVoice3(model_dir), no manual assembly needed). Same repo-naming convention as the earlier
en-UG/ber-MA/umb-AO/wo-SN push (push_hf_bundles.py).

Recent-only (not "true best across archived+current" like generate_best_all_languages.py) is
deliberate here: archived checkpoints predate the 2026-08-22 tokenizer vocab expansion (277
added tokens), and every bundle -- including these -- symlinks/copies the CURRENT
CosyVoice-BlankEN tokenizer files, so an archived-trained llm would have a stale mapping for
<|endofprompt|> and CosyVoice3's other special tokens (see conversation). These three
languages' recent llm+flow pairs were already synthesized and confirmed sounding normal
(ha-NG 14.48s/peak 0.37, ig-NG 10.52s/peak 0.84, tw-GH 13.04s/peak 0.80) in the
archived-vs-recent comparison run, hence "proven".
"""
import argparse
import glob
import os
import shutil

import torch
from huggingface_hub import HfApi

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
CURRENT_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual"
BUNDLE_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/hf_bundles_proven"

ORG = "all-lab"
# Recent llm+flow already synthesized and confirmed sounding normal for each of these in the
# archived-vs-recent comparison run: ha-NG 14.48s/peak 0.37, tw-GH 13.04s/peak 0.80,
# ig-NG 10.52s/peak 0.84, ee-GH 8.04s/peak 0.24 -- "proven" for all four.
LANGS = ["ha-NG", "tw-GH", "ig-NG", "ee-GH"]
COLLECTION_TITLE = "CosyVoice3 TTS — African Language Fine-tunes"


def highest_checkpoint(stage_dir):
    files = [p for p in glob.glob(f"{stage_dir}/*.pt") if os.path.basename(p) != "init.pt"]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def best_checkpoint(stage_dir):
    tb_dir = stage_dir.replace("/llm", "/tensorboard/llm").replace("/flow", "/tensorboard/flow")
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(tb_dir, size_guidance={"scalars": 0})
        ea.Reload()
        events = ea.Scalars("CV/loss")
        best_event = min(events, key=lambda e: e.value)
        files = [p for p in glob.glob(f"{stage_dir}/*.pt") if os.path.basename(p) != "init.pt"]
        if not files:
            return None
        return min(files, key=lambda p: abs(os.path.getmtime(p) - best_event.wall_time))
    except Exception:
        return None


def resolve_recent(stage_dir):
    return best_checkpoint(stage_dir) or highest_checkpoint(stage_dir)


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def build_bundle(lang):
    bundle_dir = f"{BUNDLE_ROOT}/{lang}"
    os.makedirs(bundle_dir, exist_ok=True)

    llm_ckpt = resolve_recent(f"{CURRENT_ROOT}/individual_{lang}/llm")
    flow_ckpt = resolve_recent(f"{CURRENT_ROOT}/individual_{lang}/flow")
    assert llm_ckpt is not None, f"{lang}: no recent llm checkpoint found"
    assert flow_ckpt is not None, f"{lang}: no recent flow checkpoint found"
    print(f"[{lang}] llm={llm_ckpt}")
    print(f"[{lang}] flow={flow_ckpt}")

    print(f"[{lang}] cleaning llm/flow (tensor-only state dicts)...")
    clean(llm_ckpt, f"{bundle_dir}/llm.pt")
    clean(flow_ckpt, f"{bundle_dir}/flow.pt")

    print(f"[{lang}] copying shared assets from pretrained release (original vocoder)...")
    for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]:
        shutil.copy2(f"{PRETRAINED_DIR}/{asset}", f"{bundle_dir}/{asset}")
    blank_en_dst = f"{bundle_dir}/CosyVoice-BlankEN"
    if os.path.exists(blank_en_dst):
        shutil.rmtree(blank_en_dst)
    shutil.copytree(f"{PRETRAINED_DIR}/CosyVoice-BlankEN", blank_en_dst)

    return bundle_dir


def push(api, lang, bundle_dir, coll):
    repo_id = f"{ORG}/cosyvoice3-individual-{lang}"
    print(f"[upload] {repo_id} <- {bundle_dir}")
    api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
    api.upload_folder(repo_id=repo_id, folder_path=bundle_dir, repo_type="model")
    if coll is not None:
        api.add_collection_item(coll.slug, item_id=repo_id, item_type="model", exists_ok=True)
    print(f"[done] {repo_id}")
    return repo_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=LANGS, help="subset of LANGS to build+push, e.g. --langs ee-GH")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN_ORGOWNER") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    print("authenticated as:", api.whoami().get("name"))

    collections = list(api.list_collections(owner=ORG))
    coll = next((c for c in collections if c.title == COLLECTION_TITLE), None)
    if coll is None:
        print(f"WARNING: collection '{COLLECTION_TITLE}' not found under {ORG}, repos will be created without adding to a collection")

    pushed = {}
    for lang in args.langs:
        bundle_dir = build_bundle(lang)
        repo_id = push(api, lang, bundle_dir, coll)
        pushed[lang] = repo_id

    print("\n=== all pushed ===")
    for lang, repo_id in pushed.items():
        print(f"  {lang}: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
