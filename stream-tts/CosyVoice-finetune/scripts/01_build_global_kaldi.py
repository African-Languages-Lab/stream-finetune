"""
Builds ONE global, deduplicated Kaldi-style data dir (per split) from all 41 Nemotron
manifests, which downstream feature extraction (speaker embedding + speech token) runs
against exactly once. Individual/cluster/combined regimes are later built by slicing this
global set -- re-extracting features per regime would multiply an already very expensive
step by 47x, since most of the underlying audio is shared across regimes (a language's
audio is a subset of its cluster's and of the combined pool; the deliberate English<->local
-language duplication from the Nemotron manifests means the SAME audio file can legitimately
appear under multiple target_lang codes, but its speech-derived features only need computing
once).

utt_id is a stable md5 hash of the audio path, so any downstream script can independently
recompute the same id from a manifest row without needing a persisted lookup table.

CosyVoice's speech tokenizer (tools/extract_speech_token.py) hard-caps at 30s and silently
emits an empty token list above that (see single_job: 'do not support extract speech token
for audio longer than 30s') rather than raising -- so utterances over 30s are filtered here,
otherwise they'd silently produce corrupt (empty) training examples downstream.
"""
import hashlib
import json
from pathlib import Path

NEMOTRON_MANIFEST_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/manifests_clean")
OUT_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data/global")

MIN_DURATION = 0.5
MAX_DURATION = 29.9  # CosyVoice's speech tokenizer hard-caps at 30s

# Keeps CosyVoice3's instruct/emotion-style conditioning pathway exercised during SFT --
# without it, that channel gets zero training signal and risks drifting from the base
# model's instruction-following behavior. Our source data has no per-utterance emotion
# labels to condition on for real, so this mirrors the reference recipe's own constant
# placeholder rather than claiming emotional diversity we don't actually have.
INSTRUCT_TEXT = "You are a helpful assistant.<|endofprompt|>"


def utt_id_for(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode()).hexdigest()[:16]


def pseudo_speaker_for(audio_path: str) -> str:
    # Source dataset folder name, e.g. .../audio/ghana-english-asr-2700hrs/xyz.wav ->
    # "ghana-english-asr-2700hrs". No real speaker labels exist in this data; grouping by
    # source corpus is a reasonable proxy for CosyVoice's spk2embedding averaging.
    return Path(audio_path).parent.name


def build_split(split: str):
    manifests = sorted(NEMOTRON_MANIFEST_DIR.glob(f"*_{split}.jsonl"))
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    n_rows, n_kept, n_filtered_dur, n_dup = 0, 0, 0, 0

    for mpath in manifests:
        with open(mpath) as f:
            for line in f:
                row = json.loads(line)
                n_rows += 1
                dur = row["duration"]
                if dur < MIN_DURATION or dur > MAX_DURATION:
                    n_filtered_dur += 1
                    continue
                audio_path = row["audio_filepath"]
                utt = utt_id_for(audio_path)
                if utt in utt2wav:
                    n_dup += 1
                    continue  # already registered from another language's manifest
                utt2wav[utt] = audio_path
                utt2text[utt] = row["text"].replace("\n", " ").replace("\t", " ").strip()
                spk = pseudo_speaker_for(audio_path)
                utt2spk[utt] = spk
                spk2utt.setdefault(spk, []).append(utt)
                n_kept += 1

    out = OUT_DIR / split
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "wav.scp", "w") as f:
        for utt, wav in utt2wav.items():
            f.write(f"{utt} {wav}\n")
    with open(out / "text", "w") as f:
        for utt, text in utt2text.items():
            f.write(f"{utt} {text}\n")
    with open(out / "utt2spk", "w") as f:
        for utt, spk in utt2spk.items():
            f.write(f"{utt} {spk}\n")
    with open(out / "spk2utt", "w") as f:
        for spk, utts in spk2utt.items():
            f.write(f"{spk} {' '.join(utts)}\n")
    with open(out / "instruct", "w") as f:
        for utt in utt2text:
            f.write(f"{utt} {INSTRUCT_TEXT}\n")

    print(f"[{split}] rows={n_rows} kept={n_kept} filtered_duration={n_filtered_dur} "
          f"deduped={n_dup} speakers={len(spk2utt)}")
    return n_kept


def main():
    total = 0
    for split in ("train", "dev"):
        total += build_split(split)
    print(f"\nTotal unique utterances across all splits: {total}")


if __name__ == "__main__":
    main()
