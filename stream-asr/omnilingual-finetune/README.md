# Omnilingual ASR — fine-tuning

> **Superseded in part — see [TRAINING_PATH.md](TRAINING_PATH.md).** The CTC + tsv/wrd setup below cannot condition on language (the CTC family has no language input, and `ManifestStorage` has no language column). The training path uses the LLM family with parquet data.


Mirrors `NeMo-finetune/`: `scripts/` builds data, `configs/` holds the recipe, `runs/` holds
the sbatch launchers. The upstream code is cloned unmodified in `../omnilingual-asr`.

## What the plan runs into

**One model conditioned on 38 languages: yes.** Omnilingual conditions its decoder on a
language token and ships 1668 of them. 33 of our 38 map directly; `scripts/lang_map.py`
holds the mapping and every entry was checked against `lang_ids.py`.

**Unlimited audio length: not together with fine-tuning.** The `omniASR_LLM_Unlimited_*_v2`
models decode arbitrarily long audio, but the upstream README states plainly that
"finetuning recipes for this model are currently not supported". So the two halves of the
plan pull apart, and the choice is:

- fine-tune a limited-length model (≈30 s cap) and segment long audio at inference, or
- run an Unlimited model as-is, unfine-tuned, and accept its stock accuracy on our
  languages.

Segmenting is the usual answer for ASR and costs little: audio is split on silence, each
piece transcribed, the text joined. It is not the same problem as TTS streaming, where a
seam is audible — here a seam is just a sentence boundary.

**Five languages have no token yet**, listed in `lang_map.UNSUPPORTED`:

| ours | why |
|---|---|
| sesotho | `sot_*` absent. `nso_Latn` is Northern Sotho, which is **sepedi** — reusing it would train sesotho as sepedi |
| ndebele | `nbl`/`nde`/`ndc` all absent; `zul_Latn` is the closest relative |
| swati | `ssw` absent; `zul_Latn` is the closest relative |
| venda | `ven` absent; no close relative |
| berber | `tzm` absent, but `kab_Latn`, `shi_Latn`, `rif_Latn`, `rif_Arab`, `taq_Latn` exist — pick the variety our corpus holds |

Omnilingual's own claim is that a language can be added from a few paired examples, so these
are candidates for that rather than losses. They are deliberately left unmapped so nothing
trains onto a wrong token by accident.

## English contamination

English is filed under languages that are not English, in two different ways, and both had
to be removed before any of this trains.

**Whole datasets under the wrong language.** `ghana-english-asr-2700hrs` sits in ewe (99.5 h)
and twi (237.4 h) as well as its own `en-GH`. `afrispeech-200` — which is African-accented
**English** — sits in ten languages. The Nigerian English crowdsourced set sits in hausa,
igbo and yoruba. Before filtering, 64 % of our ewe hours and 48 % of our twi hours were
English audio that would have trained under `ewe_Latn` and `aka_Latn`.

**Single English utterances inside native datasets.** The South African NCHLT `aux` sets are
the worst: 10–25 % of `nchlt-sot-aux1`, `nchlt-ssw-aux1` and `nchlt-tsn-aux1` are English
phrases — "united states senate", "central processing unit", "narrow gauge railway" — read
by Sotho/Swati/Tswana speakers and filed under the Bantu language. A blocklist of dataset
names cannot catch these.

So every transcript is classified with fastText LID (`facebook/fasttext-language-identification`,
the 218-label NLLB model) and the rows it calls English are dropped from every language whose
token is not `eng_Latn`.

### Code-switching is kept

Omnilingual has no mixed-language mode. `create_lang_inputs()` prepends exactly **one**
language id per utterance, so the model can never be *told* that a speaker switched. But the
id is a hint, not a gate: `lang_embeddings_p = 0.5` on the 300M/1B/3B architectures means
half of all training examples are trained with the id zeroed, and the output vocabulary is
shared across all 1668 languages. Nothing stops the model emitting English words while
conditioned on `hau_Latn` — it has to have learned to. **Code-switching is therefore purely a
property of the training data**, and filtering it out is how you get a model that cannot do it.

Top-1 English cannot tell the two apart. "Mo ti book appointment fun Monday morning" is a
Yoruba speaker code-switching and comes back `eng_Latn` at p=0.92. What separates them is
whether the native token survives anywhere in the top-5:

| utterance | top-1 | native token in top-5 | |
|---|---|---|---|
| Mo ti book appointment fun Monday morning | eng 0.92 | `yor_Latn` 0.06 | kept |
| Doctor din ya ce mu monitor da shi kafin surgery | `hau_Latn` 1.00 | — | never flagged |
| Me kɔɔ hospital no na doctor no kaa sɛ menya rest | `twi_Latn` 0.98 | — | never flagged |
| united states senate | eng 1.00 | no `sot_Latn` at any rank | dropped |
| central processing unit | eng 0.95 | no `tsn_Latn` at any rank | dropped |
| He was an enthusiast and a desert dweller. | eng 1.00 | no `afr_Latn` at any rank | dropped |

So a row is called English only when English wins **and** the native language has been pushed
out of the top-5 entirely. Pass `--drop-code-switched` to turn the exemption off.

One more tier on top of that: the exemption is for English appearing inside native data, not
native words appearing inside English data. `afrispeech-200` is African-accented English, and
the single row the code-switch test wanted to keep there was "Plan to check Lytes
Ifechukwukwuru BUN/Cr CBC" — English carrying an Igbo name. So once a source under a
non-English language comes back over 60 % English, the whole source goes, exemption included.

**The filesystem says the same thing independently.** The audio these manifests point at is
physically stored under `speech_out/english/` — 53.2 % of our twi hours and 55.6 % of our ewe
hours resolve to files in the English tree, along with 6.4 % of yoruba, 4.5 % of zulu and
4.2 % of igbo. That is a corroborating signal that owes nothing to the language model: the
manifest builder was pointing language rows at English audio files.

**Why this does not eat the native data.** Measured before trusting it: native sources come
back at 0.0–0.2 % English (Waxal ewe 0.0 %, `Naija_Voices_Igbo` 0.2 %, `akuapem-twi-dataset`
0.0 %) while the planted ones come back at 85–100 %. The separation is not marginal, and the
rows it keeps in the NCHLT aux sets are visibly native.

**Krio is the trap, and a threshold does not fix it.** Krio is an English-lexified creole and
the model has no `kri_Latn` label, so on principle it cannot separate Krio from English. A
first pass held label-less languages to a stricter 0.90 confidence and it *still* deleted

> "A dɔn was mi fut," — "Aya, a wish in lɛf an de ɔnda mi ed"

both plain Krio, both called English above p=0.98. No threshold separates a language the
model was never taught, and picking a higher number would only have hidden that.

So for any language whose token is absent from the label set, per-row judgment is switched
off entirely and only the source-level rule applies. An aggregate rate over a whole dataset
stays meaningful when individual verdicts do not — a wholesale-English dataset misfiled under
Krio would still be caught — while no individual Krio utterance is ever deleted on a call the
model is incapable of making. Krio now loses **zero** utterances. Venda and ndebele are
treated the same way.

`LID_ALIAS` handles the near-miss: our token for Oromo is `orm_Latn` and the model's label is
`gaz_Latn` (West Central Oromo). Without the alias Oromo would fall into the no-label path and
lose the native-token test for no reason.

The decision is written out per language rather than applied in place, so you can read
exactly which utterances were removed and why without re-running the model.

## Pipeline

```
scripts/lang_map.py          our corpus codes -> omnilingual language tokens
scripts/lid_filter.py        find English inside the non-English languages
scripts/build_manifests.py   NeMo JSONL manifests -> fairseq2 tsv/wrd/lang
configs/                     the finetune recipe
runs/setup_env.sbatch        build the venv (needs internet, so not on a compute node)
runs/run_lid_filter.sbatch   the LID pass (CPU, serial partition)
runs/run_finetune.sbatch     4-GPU training run
```

Data is **converted, not rebuilt**. The corpus was already walked when the NeMo manifests
were made, and those carry audio path, duration, transcript and language. Re-walking
`speech_out` would risk the two pipelines disagreeing about what is in the training set.

Two details that would silently corrupt training if got wrong, both handled in
`build_manifests.py`:

- **Column two of the tsv is a frame count, not seconds.** The reader treats it as samples;
  writing seconds makes every utterance look like a fraction of a second and the batcher
  mis-buckets everything.
- **A parallel `.lang` file.** The stock recipe has no place for a per-example language
  token, and one combined model over 38 languages needs one. Keeping it beside the tsv/wrd
  pair leaves the stock reader untouched.

## Running it

```bash
sbatch runs/setup_env.sbatch                      # once
sbatch runs/run_lid_filter.sbatch                 # once, writes <lang>.english.tsv

python scripts/build_manifests.py \
  --out /leonardo_scratch/large/userexternal/atsado00/omni_ft/manifests \
  --english-drops /leonardo_scratch/large/userexternal/atsado00/omni_ft/english_drops

sbatch runs/run_finetune.sbatch
```

`--english-drops` is deliberately not a default: without it the build prints a warning and
keeps the English in, so the filtering is always an explicit choice you can see in the log.

Verified on a dev-split subset: 4,653 utterances over hausa/igbo/yoruba, 11.7 h, tsv/wrd/lang
aligned line for line, and the audio paths resolve.

## Model choice

Starting at `omniASR_CTC_300M`. The recipe README and section 5.7.5 of the paper both report
that fine-tuned smaller CTC checkpoints match the 7B LLM model on the languages they were
tuned for — which is this case exactly: 38 languages we hold data for, not 1600 we do not.
1B/3B are a one-line change once a 300M run trains cleanly.
