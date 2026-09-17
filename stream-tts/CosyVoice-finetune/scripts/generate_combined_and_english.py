"""Combined arm for all six languages, plus the English individual clip.

generate_best_all_languages.py covers the INDIVIDUAL arm only, and it skips English because
individual_english is a merged regime with no kaldi_data directory of its own -- its shard
lists point into the accent corpora. This fills both gaps so the set is complete:

  combined    ha-NG, ig-NG, tw-GH, ee-GH, yo-NG, english
  individual  english only (the other five come from the main script)

Every clip reuses the EXACT prompt wav, prompt transcript and target sentence the individual
run selected for that language, read back from its manifest. That is what makes individual vs
combined a real comparison rather than two different recordings: same reference speaker, same
words, only the model changes.

English has no dev data under individual_english, so its prompt/target is drawn from
individual_en-UG -- the one English regime with a usable dev set (101 entries, wavs present) --
using the original script's own pick_prompt_and_target, unmodified.

Hyperparameters are stock throughout: default CosyVoice3 calls, upstream n_timesteps, original
pretrained vocoder, nothing monkeypatched. Both inference modes are rendered per language.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torchaudio  # noqa: E402

import generate_best_all_languages as G  # noqa: E402
from generate_best_all_languages import CosyVoice3  # noqa: E402

OUT_DIR = G.OUT_DIR
WORK_ROOT = G.WORK_ROOT
COMBINED_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/combined/combined"
# the combined checkpoints used in the first vetting set: lowest-CV llm and lowest-CV flow
COMBINED_LLM = f"{COMBINED_ROOT}/llm/epoch_11_whole.pt"
COMBINED_FLOW = f"{COMBINED_ROOT}/flow/epoch_15_step_320000.pt"

LANGS = ["ha-NG", "ig-NG", "tw-GH", "ee-GH", "yo-NG", "english"]
# english borrows its prompt/target from the one English regime that has dev data
PROMPT_SOURCE = {"english": "en-UG"}


def prompt_for(code, indiv_manifest):
    """Reuse whatever the individual run picked; fall back to picking it the same way."""
    entry = indiv_manifest.get(code) or {}
    if entry.get("prompt_wav") and entry.get("target_text"):
        return {"prompt_wav": entry["prompt_wav"], "prompt_text": entry["prompt_text"],
                "target_text": entry["target_text"], "source": "individual_manifest"}
    src = PROMPT_SOURCE.get(code, code)
    pt = G.pick_prompt_and_target(src)
    if pt is None:
        return None
    pt["source"] = f"kaldi:{src}"
    return pt


def render(bundle_dir, pt, prefix, manifest):
    res = G.synthesize(bundle_dir, pt["prompt_wav"], pt["prompt_text"], pt["target_text"],
                       f"{OUT_DIR}/{prefix}_best.wav")
    for mode, r in (res.get("modes") or {}).items():
        state = f"{r.get('duration', 0):.2f}s peak={r.get('peak', 0):.4f}" if r.get("ok") \
            else f"FAILED {str(r.get('error'))[:60]}"
        print(f"    {mode:14} {state}", flush=True)
    manifest[prefix] = {"prompt_source": pt["source"], "prompt_wav": pt["prompt_wav"],
                        "prompt_text": pt["prompt_text"], "target_text": pt["target_text"],
                        "result": res}
    return res


def main():
    indiv_manifest = {}
    if os.path.exists(G.MANIFEST_PATH):
        try:
            indiv_manifest = json.load(open(G.MANIFEST_PATH))
        except Exception as e:
            print(f"could not read individual manifest ({e}); picking prompts fresh", flush=True)

    manifest = {}
    out_path = f"{OUT_DIR}/manifest_combined_and_english.json"

    # ---- combined arm, one bundle reused for every language ----
    print(f"=== building combined bundle ===\n  llm={COMBINED_LLM}\n  flow={COMBINED_FLOW}",
          flush=True)
    combined_dir = f"{WORK_ROOT}/combined"
    G.build_bundle(combined_dir, COMBINED_LLM, COMBINED_FLOW)

    for code in LANGS:
        print(f"\n=== [combined/{code}] ===", flush=True)
        pt = prompt_for(code, indiv_manifest)
        if pt is None:
            print("    no usable prompt/target, skipping", flush=True)
            manifest[f"combined_{code}"] = {"status": "skipped_no_data"}
            continue
        print(f"    prompt={os.path.basename(pt['prompt_wav'])} ({pt['source']})", flush=True)
        print(f"    target={pt['target_text'][:90]}", flush=True)
        render(combined_dir, pt, f"combined_{code}", manifest)
        json.dump(manifest, open(out_path, "w"), indent=2, ensure_ascii=False)

    # ---- individual arm: english only, the one the main script skips ----
    print("\n=== [individual/english] ===", flush=True)
    llm_ckpt, llm_src, llm_cv = G.resolve_llm("english")
    flow_ckpt, flow_src, flow_cv = G.resolve_flow("english")
    if llm_ckpt and flow_ckpt:
        print(f"    llm ={llm_ckpt} ({llm_src}, cv={llm_cv})", flush=True)
        print(f"    flow={flow_ckpt} ({flow_src}, cv={flow_cv})", flush=True)
        pt = prompt_for("english", {})
        if pt is None:
            print("    no usable prompt/target, skipping", flush=True)
        else:
            print(f"    prompt={os.path.basename(pt['prompt_wav'])} ({pt['source']})", flush=True)
            print(f"    target={pt['target_text'][:90]}", flush=True)
            d = f"{WORK_ROOT}/english_individual"
            G.build_bundle(d, llm_ckpt, flow_ckpt)
            e = render(d, pt, "english", manifest)
            manifest["english"]["llm_checkpoint"] = llm_ckpt
            manifest["english"]["flow_checkpoint"] = flow_ckpt
    else:
        print("    no usable english checkpoints, skipping", flush=True)

    json.dump(manifest, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== done; manifest -> {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
