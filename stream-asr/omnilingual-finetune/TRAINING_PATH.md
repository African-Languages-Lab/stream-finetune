# Omnilingual ASR — training path

**Goal:** one `omniASR_LLM_1B_v2` model, fine-tuned on our data, conditionable on all 38 languages.
This track is independent of the Nemotron track (`../NeMo-finetune/TRAINING_PATH.md`).

Each stage has an exit check. Do not start a stage until the previous one passes.

## Where we are

| Done | Not done |
|---|---|
| English removed from non-English languages (`omni_ft/english_drops/`, 30+ languages) | Training environment (fairseq2 needs `libsndfile`) |
| Language codes: 34 existing + 4 new (`scripts/lang_map.py`) | Parquet dataset with a language column |
| Tokenizer check: `omniASR_tokenizer_written_v2` covers all 38 languages | Word-boundary cutting of audio > 60 s |
| Per-language hours, CER baselines from the paper | Lookup-table rows for the 4 new codes |
| | Per-language dev/test sets and evaluation harness |

## O0 — Environment

- uv venv, Python 3.12: `omnilingual-asr` (fairseq2 0.6), `torch==2.8.0`, `torchaudio==2.8.0`.
  A test venv exists at `omni_ft/venv_test`; it fails only on `libsndfile`.
- Provide `libsndfile` (Leonardo module or conda-forge) and put it on `LD_LIBRARY_PATH` in the job scripts.
- Compute nodes are offline. On a login node, pre-download to the fairseq2 asset cache:
  `omniASR_LLM_1B_v2`, `omniASR_LLM_Unlimited_1B_v2`, `omniASR_CTC_1B_v2`, `omniASR_tokenizer_written_v2`.

**Exit:** on a GPU node, `ASRInferencePipeline("omniASR_LLM_1B_v2").transcribe([clip], lang=["som_Latn"])` returns text.

## O1 — Data (no rows dropped for length)

1. **Source:** `nemotron_ft/manifests_clean/*_{train,dev,test}.jsonl`, minus the rows in
   `omni_ft/english_drops/<lang>.english.tsv`.
2. **Cleaning** — every rejection counted per language and source:
   - Kikuyu: map encoding corruption (`©`, `¨`, combining `͂`) to `ĩ`/`ũ` only where certain; reject the row otherwise.
   - Berber: exclude `Tamazight-Speech-to-Arabic-Text` (28 h of Arabic *translations*, not Berber transcripts) — **pending your decision**.
   - Reject transcripts with ≥ 15 % characters unknown to `omniASR_tokenizer_written_v2` (the paper's own filter, §3.4).
   - Unicode NFC.
3. **Text style:** one style for all languages, chosen at O2 from the model's own output
   (the v2 tokenizer includes case and punctuation).
4. **Duration:**
   - ≤ 60 s: kept whole (`max_audio_len: 960_000`).
   - \> 60 s (≈ 1,050 h — mostly somali, kikuyu, zulu, arabic): cut at word boundaries into ≤ 30 s pieces,
     using forced alignment on `omniASR_CTC_1B_v2` emissions (`torchaudio.functional.forced_align`).
   - Nothing is removed. If O2 selects the Unlimited model, skip the cutting and store word timings instead.
5. **Splits:** per-language dev and test, speaker/source-disjoint, frozen once built.
   Record which sources Meta also trained on (African Next Voices, NaijaVoices, Common Voice, FLEURS train) —
   baselines on those sources will look better than they are.
6. **Parquet:** `corpus=<source>/split=<split>/language=<token>/part-*.parquet`; columns `text`,
   `audio_bytes` (FLAC, 16 kHz mono), `audio_size`; `row_group_size=100`.
   Stats TSV via `workflows/dataprep/hf_dataset_ingestion_example.py compute_stats`;
   asset card `src/omnilingual_asr/cards/datasets/all_lab_38.yaml`.
7. **New codes** (`lang_map.NEW_LANG_CODES`): append `sot_Latn`, `nbl_Latn`, `ssw_Latn`, `ven_Latn` to
   `languges_lookup_table.parquet`; grow the checkpoint's language embedding 1,694 → 1,698 rows,
   initialising the new rows from `nso_Latn`, `zul_Latn`, `zul_Latn`, `tso_Latn`.

**Exit:** `workflows/dataprep/dataloader_example.py` iterates train and dev for every language;
hours per language match the manifest totals minus the logged rejections.

## O2 — Baselines and variant decision

Zero-shot on every language's dev set, normalized CER and WER:

| model | with language code | without |
|---|---|---|
| `omniASR_LLM_1B_v2` | ✓ | ✓ |
| `omniASR_LLM_Unlimited_1B_v2` | ✓ | — |
| `omniASR_CTC_1B_v2` | — | ✓ |

**Decide:** limited vs Unlimited, and transcript style.
If Unlimited: produce `word_duration` / `text_words_merged` for every row and patch the dataset to carry them
into `batch.example` (nothing in the released repo does this).

## O3 — Pilot (1 node, 4 × A100, ~2,000 steps)

Start from `workflows/recipes/wav2vec2/asr/configs/llm-finetune.yaml`:

```yaml
model:   { name: omniASR_LLM_1B_v2 }
dataset:
  name: all_lab_38
  storage_mode: MIXTURE_PARQUET
  mixture_parquet_storage_config: { beta_corpus: 0.5, beta_language: 0.25 }   # paper's final setting
  asr_task_config: { max_audio_len: 960_000, max_num_elements: <from memory test> }
tokenizer: { name: omniASR_tokenizer_written_v2 }
optimizer: { config: { lr: 5e-05 } }        # second pilot at 1e-05
trainer:   { data_parallelism: fsdp, mixed_precision: { dtype: torch.bfloat16 } }
```

Language conditioning comes from the architecture (language code kept on 50 % of samples).

**Exit:** loss falls in both pilots; no language's dev CER worse than its O2 baseline; throughput recorded.

## O4 — Main run

- 4–8 nodes (16–32 A100), FSDP, the better pilot LR.
- Step budget from pilot throughput. At the paper's effective batch (2.1 h of audio per step), 25,000 h ≈ 12,000 steps per epoch.
- Validate and checkpoint every 1,000 steps on the per-language dev sets.

## O5 — Selection and evaluation

- Pick the checkpoint with the best macro-average dev CER across languages.
- Run test once. Report per language, with and without the language code, against O2.

## Open decisions (defaults in bold)

- Model size: **1B** (300M if memory or time is short).
- Berber translation source: **exclude**.
- Transcript style: decided at O2.
- Limited vs Unlimited: decided at O2.
