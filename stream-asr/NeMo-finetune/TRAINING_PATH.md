# Nemotron 3.5 ASR — training path

**Goal:** one cache-aware streaming model (`nvidia/nemotron-3.5-asr-streaming-0.6b`) for all 38 languages,
selected at runtime through its language slots. Independent of the Omnilingual track.

Each stage has an exit check. Do not start a stage until the previous one passes.

## What the August runs got wrong, and what replaces it

| August | This path |
|---|---|
| New 32k tokenizer, decoder + joint rebuilt from random, encoder trained from step 0 | Merged tokenizer; old decoder/joint rows copied; encoder frozen in stage A |
| No `<xx-XX>` tags in transcripts | Tag after every sentence-final punctuation mark (NVIDIA's recipe) |
| Extended language dictionary never saved into the checkpoints | Dictionary written into the model config before training |
| Validation: 200 utterances over 33 languages, unnormalized | Per-language dev sets, normalized WER/CER |
| `max_duration` 10–20 s silently dropped long audio | Nothing dropped: long audio cut at word boundaries |

## Where we are

| Done | Not done |
|---|---|
| English removed from non-English languages | NeMo clone update (67 commits behind) |
| Language slots: 13 reused, 24 new, 5 aliases (`scripts/prep_prompt_dict.py` → `prompt_dictionary_v2.yaml`) | Merged tokenizer |
| Diagnosis of the August failure | Checkpoint surgery (copy old rows) |
| | Transcript tags, word-boundary cutting, tarred shards |
| | Per-language dev sets and streaming evaluation harness |

## N0 — Environment

- `nemo_venv`. Update the NeMo clone to `origin/main` (adds the multilingual cache-aware inference pipeline,
  float32 prompt projection at inference, and training speedups). Re-apply our three local patches:
  OOMptimizer prompt schema, `tar_utils` Python < 3.12, cuFFT back-off.
- Install `torchaudio` (matched to the venv's torch) and `uroman` for the forced aligner (N1). Neither is present now.

**Exit:** the base `.nemo` transcribes one English FLEURS clip correctly with
`speech_to_text_cache_aware_streaming_infer.py target_lang=en-US att_context_size=[56,13]`.

## N1 — Data (no rows dropped)

1. **Source:** `nemotron_ft/manifests_clean`, minus `omni_ft/english_drops/<lang>.english.tsv`
   (the filter output is a list of audio paths; it is not tied to Omnilingual).
2. **Cleaning:** Kikuyu corruption mapping, Berber translation source (pending decision),
   CSV-concatenation and alignment-dump rows (already removed by `clean_manifests.py`).
   After N2, reject rows with any character the merged tokenizer cannot represent.
3. **Text:** cased and punctuated, matching the base model. Append the slot's tag after every `.`, `?`, `!`,
   and once at the end if missing — e.g. `Me kɔɔ hospital no. <tw-GH>`.
   English corpora are tagged `<en-US>` (they alias slot 0).
4. **Duration:**
   - ≤ 40 s: kept whole. The base config trained to 39.99 s; the RNNT joint costs time × tokens × vocabulary per utterance, so batch sizes come from OOMptimizer.
   - \> 40 s: cut at word boundaries into ≤ 20 s pieces using `torchaudio.pipelines.MMS_FA`
     (romanized, language-independent). Nothing is removed.
5. **Shards:** tarred per language (`type: nemo_tarred`), ~200 utterances per shard manifest;
   Lhotse `input_cfg` with per-language temperature weights (start α = 0.3, as in August).
6. **Validation:** one dev manifest per language, loaded as separate validation dataloaders, static `batch_size`,
   ≥ 50 utterances per language per validation pass.

**Exit:** shard/manifest integrity check passes (every key present, counts match); hours per language
reconciled with the logged rejections.

## N2 — Merged tokenizer

- Train a SentencePiece unigram tokenizer on our **train** text only, `--spe_character_coverage=1.0`,
  `--spe_user_defined_symbols` = every tag in `prompt_dictionary_v2.yaml`. Start at 8,000 pieces
  (trade-off: more pieces → shorter sequences but a larger joint).
- Merge into the base 13,087-piece tokenizer with NVIDIA's merge
  (`nvidia-riva/tutorials/asr-extend-tokenizer-to-newlang-ft-acoustic-model.ipynb`):
  base pieces keep their IDs, new pieces are appended, `vocab.txt` is the union.

**Exit:** 0 % unknown characters on every language (the base tokenizer fails at 99.7 % for Amharic, 74 % of Yoruba lines).

## N3 — Initial checkpoint (custom script)

1. Restore the base `.nemo`; `change_vocabulary` to the merged tokenizer.
2. Copy the decoder embedding rows and the joint output rows for the 13,087 old IDs and the blank symbol
   from the base weights. The blank index moves to the end of the new vocabulary. New rows keep their random init.
3. Write into the model config: `model_defaults.prompt_dictionary` from `prompt_dictionary_v2.yaml`,
   and `freeze_updates` for stage A (NeMo reads it from the model's own config, not the Hydra overrides).
4. Save `init_38lang.nemo`.

**Exit:** the English FLEURS clip transcribes as the base model does (proves the rows were copied), and
`target_lang="tw-GH"` is accepted.

## N4 — Precision check

Same clip, two different `target_lang` values, in bf16 and in float32. NVIDIA found the one-hot language signal
is rounded away in bf16 at inference. If the bf16 outputs do not change with the language, run `prompt_kernel`
in float32 during training.

## N5 — Stage A: decoder warm-up

- `examples/asr/speech_to_text_finetune.py`, `init_from_nemo_model=init_38lang.nemo`.
- Encoder frozen (`freeze_updates: {enabled: true, modules: {encoder: -1}}`).
- `default_prompt_mode: langID` — the language is always given.
- AdamW, cosine schedule, 2 % warmup; pilot LR 3e-4 vs 1e-4; ~10,000 steps; 2 nodes.
- Batch sizes from `scripts/speech_recognition/oomptimizer.py` bucket profile.

**Exit:** dev WER falls for every language; no language emits empty output (Amharic did in August).

## N6 — Stage B: full fine-tune

- Initialise from the best stage A `.nemo`; encoder unfrozen.
- `default_prompt_mode: unified`, `unified_auto_ratio: 0.2` so `target_lang=auto` also works.
- LR 1e-4 → 1e-6 cosine, 2 % warmup, bf16, keep the base model's four attention contexts
  `[[56,3],[56,0],[56,6],[56,13]]`.
- 8 nodes × 4 A100 (`boost_qos_lprod`, 4-day wall, or `normal` with resubmission).
- Step budget from stage A throughput (expect 100–200k).
- Validate every 2,000 steps per language; `save_top_k` by `val_wer`, `always_save_nemo: true`.

## N7 — Stage C: low-LR refinement (optional)

From the best stage B checkpoint, LR 1e-5 → 5e-7 (the Kikuyu/Dholuo paper's late-stage setting),
with the weakest languages upweighted.

## N8 — Selection and evaluation

- Standalone **streaming** evaluation at `[56,0]`, `[56,3]`, `[56,13]`, with the language given and with `auto`.
- Normalized WER, CER and space-free CER per language.
- Checkpoint averaging kept only if it beats the best single checkpoint.
- Export the `.nemo` (dictionary inside) and a model card.

## Open decisions (defaults in bold)

- English corpora: **alias to slot 0** (20 free slots remain for accent-specific slots).
- NVIDIA's original 40 languages: **no replay** — our data covers English and Arabic.
- New tokenizer size: **8,000 pieces**, revisited after measuring tokens per word.
- Berber translation source: **exclude**.
