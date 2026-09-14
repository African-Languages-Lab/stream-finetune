> **Superseded — see [NeMo-finetune/TRAINING_PATH.md](NeMo-finetune/TRAINING_PATH.md).** Kept for history; several assumptions here were wrong (the base tokenizer cannot represent ~12 of our languages, and transcripts need `<xx-XX>` language tags).

# Fine-tuning nvidia/nemotron-3.5-asr-streaming-0.6b for Yoruba, Igbo, Hausa, Twi, Ewe

Grounded in the actual cloned repo at `stream-asr/NeMo` (commit as of 2026-08-12) plus the live
HF model card / config. All file paths below are relative to `stream-asr/NeMo/`.

## 0. What this model actually is

- FastConformer-CacheAware encoder (24 layers, d_model=1024) + RNNT decoder, with a **language-ID
  prompt mechanism** grafted on: `nemo/collections/asr/models/rnnt_bpe_models_prompt.py`,
  class `EncDecRNNTBPEModelWithPrompt`.
- 40 trained language-locales, vocab_size 13,088, `num_prompts: 128` (confirmed from the live HF
  `config.json`).
- Verify after downloading the checkpoint (not yet confirmed remotely):
  `ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b", return_config=True).target`
  — docs table says "Hybrid" but everything else (HF class name, tutorial config) points to
  RNNT-only. Confirm before writing infra around it.

## 1. The three real workstreams

### A. Data (the actual bottleneck — this is on you, not tooling)

Need hours of speech + transcripts per language (Yoruba, Igbo, Hausa, Twi, Ewe). NeMo manifest
format, one JSON object per line:

```json
{"audio_filepath": "/data/audio/sample.wav", "duration": 3.42, "text": "transcript text", "target_lang": "yo-NG"}
```

- `target_lang` is required by this model's dataloader (`lang_field: target_lang` in the prompt
  config) and must be a key in the model's `prompt_dictionary`.
- Keep transcript style (casing, punctuation) consistent with what the base model already
  produces — don't casually lowercase/strip punctuation.
- Utilities: `nemo/collections/asr/parts/utils/manifest_utils.py` (`read_manifest`/`write_manifest`,
  use `ensure_ascii=False` to keep diacritics readable in the files).
- Prefer Lhotse dataloading (`model.train_ds.use_lhotse=true`) over the legacy manifest loader for
  bucketing at any real scale.
- **Include a "replay" slice of the existing 40 languages** in the training blend (Lhotse
  `input_cfg` with per-source `weight`) — this is NVIDIA's actual documented mitigation against
  catastrophic forgetting of existing language competence, not parameter freezing (see §C).

### B. Tokenizer (the hard, unresolved part)

Open question I could not verify remotely: does the shipped SentencePiece tokenizer have
`byte_fallback` enabled? Check locally once the checkpoint is downloaded:
```python
# asr_model.tokenizer.tokenizer is the underlying sentencepiece.SentencePieceProcessor
```
Even if byte-fallback exists, it only prevents crashes on unseen text — it gives the model no
subword structure for Yoruba/Igbo/Hausa/Twi/Ewe orthography (diacritics like ẹ ọ ṣ, tone marks).
With effectively zero vocabulary coverage of these 5 languages, expect near-total byte/char-level
fragmentation without intervention.

**Recommended path**: extend the tokenizer. Train a fresh SentencePiece model on a combined
corpus (approximation of existing 40-language text + your 5 new languages' transcripts) via
`scripts/tokenizers/process_asr_text_tokenizer.py --tokenizer=spe --spe_type=unigram`, then load
with `model.tokenizer.update_tokenizer=true`. This reinitializes the RNNT decoder+joint output
layer (their dim depends on vocab size) — the encoder still transfers, the decoder/joint train
mostly from scratch for the new distribution. When doing this:
```yaml
init_from_nemo_model: "/path/to/checkpoint.nemo"
init_from_nemo_model_exclude: [decoder, joint]
```
(`docs/source/asr/fine_tuning.rst:88-105`)

The alternative (`AggregateTokenizer`, per-language disjoint ID ranges, see
`tutorials/asr/Multilang_ASR.ipynb`) is NOT demonstrated anywhere combined with this
prompt-conditioned RNNT class in the codebase — would be unsupported territory.

### C. The training run

- Script: `examples/asr/speech_to_text_finetune.py`
- Config: `examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml`
  (this is what NVIDIA's own fine-tuning tutorial uses for this exact model)
- Init: `+init_from_nemo_model=<downloaded checkpoint>`
- **Override the LR/schedule** — the config's shipped `NoamAnnealing` + `lr: 2.0` is for
  pretraining from scratch, not fine-tuning. Use Cosine annealing, warmup ~1-2% of `max_steps`:
  - `~1e-4` for a large/mixed fine-tune
  - `~3e-5` for smaller domain adaptation (<20h of data)
  - `1e-5` or lower if a run is unstable/diverging
- **Language-ID prompt slots**: no architecture surgery needed — `num_prompts=128`, only ~40ish
  in active use, clear headroom. Verify locally whether `ha-NG`/`yo-NG`/`ig-NG` already have
  reserved (but untrained) indices in the real checkpoint's `prompt_dictionary` (the example
  config has them at 50/52/53 — confirm this matches the actual shipped checkpoint, don't assume).
  Assign fresh indices for Twi/Ewe (e.g. `tw-GH`, `ee-GH`) in the unused 104-127 range.
- **Anti-forgetting**: no working freeze/adapter path exists for this specific model class in the
  repo (`examples/asr/asr_adapters/` exists but isn't wired up for `EncDecRNNTBPEModelWithPrompt`
  anywhere found). The documented mitigation is the data replay blend from §A, a conservative LR,
  and monitoring `val_wer` on existing-language validation sets *during* the run, not just the
  5 new languages (NeMo supports multiple validation dataloaders via
  `setup_multiple_validation_data`).

## 2. Before committing to this plan — verify these locally

1. Download the checkpoint and print `model.cfg.model_defaults.prompt_dictionary` — confirm
   `ha-NG`/`yo-NG`/`ig-NG` indices and free slots for Twi/Ewe.
2. Inspect the tokenizer's `byte_fallback` flag and actual coverage of the 5 target scripts.
3. Confirm `cfg.target` (RNNT-only vs Hybrid) to resolve the docs-table discrepancy.

## 3. Execution guide

The NeMo repo ships its own maintained skill for this exact workflow:
`.claude/skills/nemo-speech-asr-finetune/` (`SKILL.md` + reference docs covering container setup,
Lhotse dataloading, tokenizer decisions, training/evaluation, and iterative refinement). Once
data collection for at least one language is underway, invoke that skill to walk through the
concrete setup rather than re-deriving it — it's NVIDIA's own playbook and matches this checkout.

## Bottom line

Tooling/architecture readiness is not the blocker — the language-ID prompt mechanism has room for
all 5 languages with zero architecture changes, and NVIDIA's own scripts/configs are a workable
starting skeleton. The two real blockers are (1) **sourcing hours of transcribed speech per
language** — nothing here can substitute for that — and (2) **the tokenizer extension**, which is
genuinely unproven for this exact model in NVIDIA's own docs/tutorials and will need careful
validation (watch decoder/joint reinit stability, watch for training divergence early).
