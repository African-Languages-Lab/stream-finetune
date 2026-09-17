"""
Transcribes 5 reference samples (English, Hausa, Igbo, Yoruba, Twi) with each language's
HIGHEST individual checkpoint, the HIGHEST combined checkpoint, and the HIGHEST checkpoint of
whichever cluster that language belongs to.

Two of these five languages' target_lang tags (en-NG, tw-GH) are NOT in the base model's
frozen prompt_dictionary (confirmed by direct inspection earlier) -- langID prompt_mode
crashes on an unregistered tag with "Unknown prompt key". Training itself never hit this
because prompt_mode=unified only calls the langID lookup ~50% of the time (unified_auto_ratio),
falling back to the language-agnostic "auto" index the rest of the time -- inference here
uses prompt_mode="auto" directly (deterministic, no crash) for the two unregistered
languages, and "langID" (explicit, deterministic) for the three that are registered.
"""
import json
import tempfile
from pathlib import Path

import soundfile as sf
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

REF_DIR = Path("/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-trial/reference-samples")
OUT_JSON = REF_DIR / "transcripts_highest_checkpoints.json"
OUT_TXT = REF_DIR / "transcripts_highest_checkpoints.txt"

CKPT_ROOT = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints")

# (ref wav, target_lang, prompt_mode, cluster name)
CASES = [
    ("english_ref.wav", "en-NG", "auto", "other_languages"),
    ("hausa_ref.wav", "ha-NG", "langID", "afroasiatic"),
    ("igbo_ref.wav", "ig-NG", "langID", "west_africa_niger_congo"),
    ("yoruba_ref.wav", "yo-NG", "langID", "west_africa_niger_congo"),
    ("twi_ref.wav", "tw-GH", "auto", "west_africa_niger_congo"),
]


def latest_checkpoint(d):
    # .nemo only gets (re-)exported when a run genuinely reaches its configured max_steps and
    # Trainer.fit() stops -- for any regime that hasn't hit its (recently raised) target yet,
    # that file is stale, left over from whatever the target used to be. Blindly preferring it
    # over newer .ckpt files silently picks a less-trained checkpoint. Compare mtimes instead
    # and take whichever is actually most recent.
    import re

    unfinished_stems = {p.name[: -len("-unfinished")] for p in d.glob("*-unfinished")}
    ckpt_candidates = [p for p in d.glob("*.ckpt") if p.name not in unfinished_stems]

    def epoch_key(p):
        m = re.search(r"epoch=(\d+)", p.name)
        return int(m.group(1)) if m else -1

    latest_ckpt = sorted(ckpt_candidates, key=epoch_key)[-1] if ckpt_candidates else None
    nemo_files = sorted(d.glob("*.nemo"))
    latest_nemo = nemo_files[-1] if nemo_files else None

    if latest_nemo and latest_ckpt:
        return latest_nemo if latest_nemo.stat().st_mtime >= latest_ckpt.stat().st_mtime else latest_ckpt
    return latest_nemo or latest_ckpt


def load_model(ckpt_path):
    if str(ckpt_path).endswith(".nemo"):
        return nemo_asr.models.ASRModel.restore_from(str(ckpt_path), map_location="cuda")
    return EncDecRNNTBPEModelWithPrompt.load_from_checkpoint(str(ckpt_path), map_location="cuda")


def transcribe(model, wav_path, target_lang, prompt_mode):
    duration = sf.info(str(wav_path)).duration
    with tempfile.TemporaryDirectory() as td:
        manifest_path = Path(td) / "manifest.json"
        with open(manifest_path, "w") as f:
            f.write(json.dumps({
                "audio_filepath": str(wav_path),
                "text": "",
                "duration": duration,
                "target_lang": target_lang,
            }) + "\n")
        dl_config = DictConfig({
            "manifest_filepath": None,
            "input_cfg": [{
                "type": "nemo",
                "manifest_filepath": str(manifest_path),
                "weight": 1.0,
                "tags": {"target_lang": target_lang, "prompt_mode": prompt_mode},
            }],
            "sample_rate": model.preprocessor._sample_rate,
            "batch_size": 1,
            "shuffle": False,
            "use_lhotse": True,
            "use_bucketing": False,
            "drop_last": False,
            "num_workers": 0,
            "pin_memory": True,
            "channel_selector": 0,
            "prompt_dictionary": model.cfg.model_defaults.get("prompt_dictionary"),
            "num_prompts": model.cfg.model_defaults.get("num_prompts", 128),
            "subsampling_factor": model.cfg.get("subsampling_factor", 8),
            "lang_field": "target_lang",
            "prompt_mode_field": "prompt_mode",
            "default_prompt_mode": prompt_mode,
        })
        dataloader = model._setup_dataloader_from_config(config=dl_config)
        hyps = model.transcribe(audio=dataloader, verbose=False)
    return hyps[0].text if hasattr(hyps[0], "text") else str(hyps[0])


def main():
    results = []
    for wav_name, target_lang, prompt_mode, cluster_name in CASES:
        wav_path = REF_DIR / wav_name
        checkpoints = {
            "individual": latest_checkpoint(CKPT_ROOT / "individual" / target_lang / f"individual_{target_lang}" / "checkpoints"),
            "cluster": latest_checkpoint(CKPT_ROOT / "clusters" / cluster_name / f"cluster_{cluster_name}" / "checkpoints"),
            "combined": latest_checkpoint(CKPT_ROOT / "combined" / "nemotron_combined_multilingual" / "checkpoints"),
        }
        for kind, ckpt_path in checkpoints.items():
            print(f"=== {wav_name} ({target_lang}) / {kind} ===", flush=True)
            if ckpt_path is None:
                print("  no checkpoint found", flush=True)
                results.append({"file": wav_name, "target_lang": target_lang, "kind": kind, "error": "no checkpoint"})
                continue
            print(f"  checkpoint: {ckpt_path}", flush=True)
            try:
                model = load_model(ckpt_path)
                model.eval()
                text = transcribe(model, wav_path, target_lang, prompt_mode)
                print(f"  -> {text}", flush=True)
                results.append({"file": wav_name, "target_lang": target_lang, "kind": kind,
                                 "checkpoint": str(ckpt_path), "prompt_mode": prompt_mode, "transcript": text})
                del model
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                results.append({"file": wav_name, "target_lang": target_lang, "kind": kind, "error": str(e)})

    with open(OUT_TXT, "w") as f:
        for r in results:
            f.write(f"[{r['target_lang']} / {r['kind']}] {r['file']}\n")
            f.write(f"  {r.get('transcript') or ('ERROR: ' + r.get('error', 'unknown'))}\n\n")
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_TXT} and {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
