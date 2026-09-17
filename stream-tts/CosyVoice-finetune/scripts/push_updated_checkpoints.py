"""Re-push the 13 language repos whose flow improved since the 24 August upload.

Only these 13 changed. The other 11 served languages (arabic, berber, lingala, luganda,
malagasy, oromo, sepedi, sesotho, tswana, umbundu, amharic) have flows that never advanced
past what is already on the Hub, so re-pushing them would upload identical weights.

    venda    +7.63%   igbo     +5.64%   zulu      +1.82%
    hausa    +7.23%   ewe      +4.40%   twi       +1.81%
    fula     +6.14%   xhosa    +2.75%   tigrinya  +1.64%
    tsonga   +5.94%   chichewa +2.13%   swahili   +1.01%
                                        somali    +0.92%

CHECKPOINT SELECTION is lowest-CV within the CURRENT run, which is deliberately the same
rule vet_six_languages.py used to produce the clips being listened to. It is NOT the
"best across current + archived" rule from generate_best_all_languages.py: that picks a
different ewe llm (archived epoch_2_whole, cv 3.3094, against current 3.3099 -- 0.015%
apart, inside noise), and shipping a checkpoint nobody has heard is the wrong trade. What
was validated by ear is what goes up.

This differs again from build_and_push_proven_hf_bundles.py, the script that created these
repos, which selects "recent only, never archived" by matching tensorboard minima to
checkpoint mtimes -- a fragile pairing with no record of what it chose.

Which is the other thing this fixes: every repo gets a README.md stating the checkpoint,
step and CV behind its weights. None of them carried any provenance, so identifying what
was deployed previously required scanning the filesystem.

Only llm.pt and flow.pt are replaced. The shared assets already in each repo
(CosyVoice-BlankEN, speech_tokenizer_v3.onnx, campplus.onnx, hift.pt, cosyvoice3.yaml) are
untouched -- they come from the pretrained model and have not changed.

Checkpoints are stripped to tensors only before upload, matching how the existing files were
produced. Each language is cleaned, uploaded, then its temp copy deleted, so peak scratch use
stays at one language rather than 43 GB.

Run:  python push_updated_checkpoints.py            # dry run, prints the plan
      python push_updated_checkpoints.py --apply    # perform the upload
"""
import json
import os
import shutil
import sys
import tempfile

import torch
from huggingface_hub import HfApi

# PUSH_PLAN selects which plan to run: the 13 improved repos, or all 38.
PLAN = os.environ.get("PUSH_PLAN", "/scratch_local/claude-131198/-leonardo-home-userexternal-atsado00-all-lab-workspace-002-all-data-stream/84e1d518-8227-454f-892d-668d323f025c/scratchpad/push_plan.json")
ORG = "all-lab"
WORK = "/leonardo_scratch/large/userexternal/atsado00/hf_push_tmp"

# HF rejects a YAML header whose `language:` is not an ISO code ("Invalid metadata in
# README.md"), which failed the first pass while the weights themselves uploaded fine.
# The header carries no benefit for a private weights repo, so it is simply omitted.
README = """# CosyVoice3 — {lang}

Fine-tuned CosyVoice3 for {lang} ({code}). Paired with the ORIGINAL pretrained vocoder
(`hift.pt`); the vocoder is never fine-tuned.

## Checkpoints in this repo

| stage | checkpoint | step | CV loss |
|---|---|---|---|
| llm  | `{llm_name}`  | {llm_step:,} | {llm_cv:.4f} |
| flow | `{flow_name}` | {flow_step:,} | {flow_cv:.4f} |

Both are the **lowest-CV checkpoint of the current training run**, not the newest. That
distinction matters: several languages diverge badly with continued training, so the newest
checkpoint is often far from the best. Igbo's final llm, for instance, sits at more than
twice its own minimum CV.

## Inference

Use `inference_zero_shot`, passing the reference clip's transcript. It conditions the LLM on
a paired example of text and speech tokens in this language, which is what makes the output
sound native.

`inference_cross_lingual` deletes `prompt_text` and `llm_prompt_speech_token` from the model
input. The voice still carries through the flow prompt and speaker embedding, but the LLM
loses its demonstration of how the language sounds, and the result is noticeably less
native. Use it only when no trustworthy transcript exists.

Note that zero-shot degrades when the text to synthesise is shorter than half the reference
transcript — CosyVoice warns about this, and generation can collapse to a fraction of a
second.

Updated {date}.
"""


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def main():
    apply = "--apply" in sys.argv
    token = os.environ.get("HF_TOKEN_ORGOWNER")
    if not token:
        raise SystemExit("set HF_TOKEN_ORGOWNER first:  set -a; . ./.env; set +a")
    api = HfApi(token=token)
    plan = json.load(open(PLAN))
    os.makedirs(WORK, exist_ok=True)

    print(f"{len(plan)} repos to update, {sum(v['gb'] for v in plan.values()):.1f} GB total\n")
    if not apply:
        for lang, v in plan.items():
            print(f"  {lang:10} llm={os.path.basename(v['llm'])[:-3]:24} cv {v['llm_cv']:.4f}"
                  f"   flow={os.path.basename(v['flow'])[:-3]:22} cv {v['flow_cv']:.4f}")
        print("\ndry run — re-run with --apply")
        return

    import datetime as dt
    today = dt.date.today().isoformat()
    done, failed = [], []
    for n, (lang, v) in enumerate(plan.items(), 1):
        repo = f"{ORG}/cosyvoice3-individual-{lang}"
        print(f"\n[{n}/{len(plan)}] {repo}", flush=True)
        d = os.path.join(WORK, lang)
        os.makedirs(d, exist_ok=True)
        try:
            print("   cleaning checkpoints...", flush=True)
            clean(v["llm"], f"{d}/llm.pt")
            clean(v["flow"], f"{d}/flow.pt")
            readme = README.format(
                lang=lang, code=v["code"], date=today,
                llm_name=os.path.basename(v["llm"])[:-3], llm_step=v["llm_step"],
                llm_cv=v["llm_cv"],
                flow_name=os.path.basename(v["flow"])[:-3], flow_step=v["flow_step"],
                flow_cv=v["flow_cv"])
            open(f"{d}/README.md", "w").write(readme)

            msg = (f"Update to lowest-CV checkpoints: llm {os.path.basename(v['llm'])[:-3]} "
                   f"(cv {v['llm_cv']:.4f}), flow {os.path.basename(v['flow'])[:-3]} "
                   f"(cv {v['flow_cv']:.4f})")
            for f in ("llm.pt", "flow.pt", "README.md"):
                print(f"   uploading {f}...", flush=True)
                api.upload_file(path_or_fileobj=f"{d}/{f}", path_in_repo=f,
                                repo_id=repo, repo_type="model", commit_message=msg)
            done.append(lang)
            print(f"   OK", flush=True)
        except Exception as e:
            failed.append((lang, str(e)[:120]))
            print(f"   FAILED: {str(e)[:160]}", flush=True)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    print(f"\n=== updated {len(done)}: {', '.join(done)}")
    if failed:
        print(f"=== FAILED {len(failed)}:")
        for l, e in failed:
            print(f"   {l}: {e}")


if __name__ == "__main__":
    main()
