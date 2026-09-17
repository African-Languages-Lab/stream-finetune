"""Push the verified per-language checkpoints to HF, one repo at a time.

Each language's checkpoint pair is the one the sweeps confirmed produces real speech -- the
LATEST epoch that works, not the best-scoring one, since the duration metric cannot separate
two working checkpoints finely enough to justify shipping an under-trained epoch over a
later one.

The vocoder is ALWAYS the original pretrained hift.pt. Every fine-tuned vocoder was judged
worse, so training it was abandoned; serving pairs a fine-tuned llm+flow with the release
vocoder.

Staging is built and deleted per language rather than all at once -- 15 languages at ~4.4 GB
each would be 66 GB of scratch held for no reason.
"""
import json
import os
import shutil
import sys

import torch
from huggingface_hub import HfApi

PRE = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual"
STAGE = "/leonardo_scratch/large/userexternal/atsado00/push_stage"
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]

ISO = {"afrikaans": "af-ZA", "bambara": "bm-ML", "bemba": "bem-ZM", "fon": "fon-BJ",
       "kanuri": "kr-NG", "kikuyu": "ki-KE", "kinyarwanda": "rw-RW", "krio": "kri-SL",
       "ndebele": "nd-ZW", "shona": "sn-ZW", "swati": "ss-SZ", "wolof": "wo-SN",
       "yoruba": "yo-NG", "english": "english"}

# language -> (llm file, flow file, serving mode)
PLAN = {
    "afrikaans":   ("epoch_32_whole.pt",  "epoch_159_whole.pt", "zero-shot"),
    "bambara":     ("epoch_9_whole.pt",   "epoch_9_whole.pt",   "zero-shot"),
    "bemba":       ("epoch_21_whole.pt",  "epoch_106_whole.pt", "cross-lingual"),
    "fon":         ("epoch_130_whole.pt", "epoch_130_whole.pt", "cross-lingual"),
    "kanuri":      ("epoch_97_whole.pt",  "epoch_97_whole.pt",  "zero-shot"),
    "kikuyu":      ("epoch_27_whole.pt",  "epoch_126_whole.pt", "cross-lingual"),
    "kinyarwanda": ("epoch_0_step_70000.pt", "epoch_3_step_70000.pt", "zero-shot"),
    "krio":        ("epoch_165_whole.pt", "epoch_165_whole.pt", "zero-shot"),
    "ndebele":     ("epoch_12_whole.pt",  "epoch_12_whole.pt",  "cross-lingual"),
    "shona":       ("epoch_54_whole.pt",  "epoch_54_whole.pt",  "zero-shot"),
    "swati":       ("epoch_16_whole.pt",  "epoch_16_whole.pt",  "zero-shot"),
    "wolof":       ("epoch_199_whole.pt", "epoch_199_whole.pt", "cross-lingual"),
    # the sweep tested flows 10/20/30 against llm epoch_5 and all three worked, so the latest
    # wins by the same prefer-later rule
    "yoruba":      ("epoch_5_whole.pt",   "epoch_30_whole.pt",  "zero-shot"),
    # the combined English model, trained on Ghanaian + Nigerian + Ugandan data pooled
    "english":     ("epoch_1_step_70000.pt", "epoch_1_step_70000.pt", "cross-lingual"),
}

CARD = """---
license: apache-2.0
tags: [text-to-speech, cosyvoice, african-languages, {lang}]
---

# CosyVoice3 — {lang}

| component | checkpoint |
|---|---|
| `llm.pt`  | `{llm}` |
| `flow.pt` | `{flow}` |
| `hift.pt` | **original pretrained** — the vocoder is never fine-tuned |

Chosen by synthesis, not by loss: the latest checkpoint that produces real speech. Earlier
verdicts for several languages came from one automatically-selected checkpoint and were wrong
-- yoruba was written off as "collapses to under a second" on a checkpoint that does exactly
that, while earlier epochs work fine.

**Serving mode: {mode}.**{note}

```python
from cosyvoice.cli.cosyvoice import CosyVoice3
model = CosyVoice3("all-lab/cosyvoice3-individual-{lang}", fp16=False)

# <|endofprompt|> ends an INSTRUCTION PREAMBLE -- the reference transcript goes AFTER it.
# Appending it instead makes the model speak the reference clip before your text.
prompt_text = "You are a helpful assistant.<|endofprompt|>" + reference_transcript
out = list(model.inference_zero_shot(text, prompt_text, "ref.wav", stream=False))

# No transcript, or one you cannot vouch for -> cross-lingual. No reference text reaches the
# model, so it cannot be spoken, and a transcript that mismatches its audio cannot collapse it.
out = list(model.inference_cross_lingual(
    "You are a helpful assistant.<|endofprompt|>" + text, "ref.wav", stream=False))
```

`inference_zero_shot` splits long input into ~80-token chunks and **yields one segment per
chunk** — concatenate them all or you return only the opening of a long passage.
Reference audio must be at least 16 kHz.
"""
XL_NOTE = ("\n\nThis language's corpus transcripts do not reliably match their audio, which "
           "collapses zero-shot generation. Cross-lingual passes no transcript at all, so it "
           "is both the working mode and the safe one.")


def main():
    dry = "--yes" not in sys.argv
    only = {a for a in sys.argv[1:] if not a.startswith("--")}
    api = HfApi(token=os.environ["HF_TOKEN_ORGOWNER"])
    os.makedirs(STAGE, exist_ok=True)
    print(f"{'language':14}{'llm':24}{'flow':24}{'mode':14}")
    for lang, (llm, flow, mode) in sorted(PLAN.items()):
        print(f"{lang:14}{llm:24}{flow:24}{mode:14}")
    if dry:
        print("\nDRY RUN -- re-run with --yes to push")
        return

    for lang, (llm, flow, mode) in sorted(PLAN.items()):
        if only and lang not in only:
            continue
        iso = ISO[lang]
        d = os.path.join(STAGE, lang)
        src_llm = f"{ROOT}/individual_{iso}/llm/{llm}"
        src_flow = f"{ROOT}/individual_{iso}/flow/{flow}"
        if not (os.path.exists(src_llm) and os.path.exists(src_flow)):
            print(f"{lang}: MISSING checkpoint, skipping ({src_llm if not os.path.exists(src_llm) else src_flow})",
                  flush=True)
            continue
        os.makedirs(d, exist_ok=True)
        try:
            for src, dst in ((src_llm, "llm.pt"), (src_flow, "flow.pt")):
                sd = torch.load(src, map_location="cpu", weights_only=True)
                t = {k: v for k, v in sd.items() if torch.is_tensor(v)}
                bad = [k for k, v in t.items()
                       if v.dtype.is_floating_point and (torch.isnan(v).any() or torch.isinf(v).any())]
                if bad:
                    raise RuntimeError(f"{dst} has NaN/Inf in {len(bad)} tensors")
                torch.save(t, os.path.join(d, dst))
            for a in SHARED:
                shutil.copy2(f"{PRE}/{a}", os.path.join(d, a))
            open(os.path.join(d, "README.md"), "w").write(
                CARD.format(lang=lang, llm=llm, flow=flow, mode=mode,
                            note=XL_NOTE if mode == "cross-lingual" else ""))
            repo = f"all-lab/cosyvoice3-individual-{lang}"
            api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
            api.upload_folder(folder_path=d, repo_id=repo, repo_type="model",
                              commit_message=f"{lang}: llm {llm} + flow {flow}, original vocoder")
            print(f"  PUSHED {repo}", flush=True)
        except Exception as e:
            print(f"  FAILED {lang}: {str(e)[:120]}", flush=True)
        finally:
            shutil.rmtree(d, ignore_errors=True)   # keep scratch flat, one language at a time


if __name__ == "__main__":
    main()
