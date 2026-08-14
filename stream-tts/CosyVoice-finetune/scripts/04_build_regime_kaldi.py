"""
Builds a Kaldi-format dir (wav.scp/text/utt2spk/spk2utt + sliced utt2embedding.pt/
spk2embedding.pt/utt2speech_token.pt) for one individual-language or cluster regime, by
reading that regime's original Nemotron manifest(s) directly and looking features up from
the already-extracted global pool (03_merge_features.py output) -- no re-extraction.

The "combined" regime needs no separate construction: it's exactly the global deduplicated
set built by 01_build_global_kaldi.py, since CosyVoice's Kaldi format has no per-row
language field the way Nemotron's target_lang-tagged manifests do, so pooling all 41
languages together IS the global set.

Same duration filter as the global build (0.5-29.9s) is re-applied per source manifest so
regime membership is computed identically; rows are then deduped by utt_id (audio-derived,
so the same physical file merged into multiple codes -- e.g. an English utterance merged
into more than one Ghanaian language in the same cluster -- collapses to one entry, which is
correct since CosyVoice's data has no per-row language tag to preserve duplicates for).
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

NEMOTRON_MANIFEST_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/manifests_clean")
GLOBAL_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data/global")
OUT_ROOT = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data")

MIN_DURATION = 0.5
MAX_DURATION = 29.9

CLUSTERS = {
    "afroasiatic": ["ar-AR", "am-ET", "ti-ER", "ha-NG", "or-KE", "so-SO", "ber-MA"],
    "east_africa_bantu": ["ki-KE", "rw-RW", "lg-UG", "sw-KE"],
    "southern_central_bantu": ["ny-MW", "ln-CD", "nd-ZW", "nso-ZA", "st-ZA", "sn-ZW", "ss-SZ",
                                "ts-ZA", "tn-BW", "umb-AO", "ve-ZA", "xh-ZA", "zu-ZA", "bem-ZM"],
    "west_africa_niger_congo": ["bm-ML", "ee-GH", "fon-BJ", "ff-SN", "ig-NG", "tw-GH", "wo-SN", "yo-NG"],
    "other_languages": ["af-ZA", "kr-NG", "mg-MG", "kri-SL", "en-GH", "en-NG", "en-ZA", "en-UG"],
}


def utt_id_for(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode()).hexdigest()[:16]


def collect_utts(target_langs, split, valid_utts):
    """Read the given target_lang codes' own manifests for this split, filtered/deduped.

    valid_utts must be the intersection of global utt2spk, utt2embedding, and
    utt2speech_token keys -- utt2spk alone isn't enough. utt2spk comes from the cleaned
    manifests (01_build_global_kaldi.py), while embeddings/tokens come from the extraction
    shards merge (03_merge_features.py), and a small number of utt_ids legitimately exist in
    one but not the other (e.g. a speech token extraction that came back empty and got
    dropped during the merge). Checking only utt2spk let such rows through into wav.scp/text
    while their embedding/token was silently absent, which crashed make_parquet_list.py with
    a bare KeyError deep inside a multiprocessing worker (confirmed: this is exactly what
    took down the first other_languages cluster build).
    """
    utt2wav, utt2text, utt2spk = {}, {}, {}
    global_utt2spk = _global_utt2spk_cache.setdefault(split, _load_global_utt2spk(split))
    for tl in target_langs:
        mpath = NEMOTRON_MANIFEST_DIR / f"{tl}_{split}.jsonl"
        if not mpath.exists():
            continue
        with open(mpath) as f:
            for line in f:
                row = json.loads(line)
                dur = row["duration"]
                if dur < MIN_DURATION or dur > MAX_DURATION:
                    continue
                audio_path = row["audio_filepath"]
                utt = utt_id_for(audio_path)
                if utt not in valid_utts:
                    continue  # missing from utt2spk, or dropped during feature extraction/merge
                if utt in utt2wav:
                    continue  # already collected (duplicate across constituent languages)
                utt2wav[utt] = audio_path
                utt2text[utt] = row["text"].replace("\n", " ").replace("\t", " ").strip()
                utt2spk[utt] = global_utt2spk[utt]
    return utt2wav, utt2text, utt2spk


_global_utt2spk_cache = {}


def _load_global_utt2spk(split):
    d = {}
    with open(GLOBAL_DIR / split / "utt2spk") as f:
        for line in f:
            utt, spk = line.rstrip("\n").split(" ", 1)
            d[utt] = spk
    return d


INSTRUCT_TEXT = "You are a helpful assistant.<|endofprompt|>"


def write_kaldi_dir(out_dir, utt2wav, utt2text, utt2spk):
    out_dir.mkdir(parents=True, exist_ok=True)
    spk2utt = {}
    for utt, spk in utt2spk.items():
        spk2utt.setdefault(spk, []).append(utt)
    with open(out_dir / "wav.scp", "w") as f:
        for utt, wav in utt2wav.items():
            f.write(f"{utt} {wav}\n")
    with open(out_dir / "text", "w") as f:
        for utt, text in utt2text.items():
            f.write(f"{utt} {text}\n")
    # Keeps CosyVoice3's instruct/emotion-style conditioning pathway exercised during SFT --
    # without it, that channel gets zero training signal and risks drifting from the base
    # model's instruction-following behavior. Our source data has no per-utterance emotion
    # labels to condition on for real, so this mirrors the reference recipe's own constant
    # placeholder rather than claiming emotional diversity we don't actually have.
    with open(out_dir / "instruct", "w") as f:
        for utt in utt2text:
            f.write(f"{utt} {INSTRUCT_TEXT}\n")
    with open(out_dir / "utt2spk", "w") as f:
        for utt, spk in utt2spk.items():
            f.write(f"{utt} {spk}\n")
    with open(out_dir / "spk2utt", "w") as f:
        for spk, utts in spk2utt.items():
            f.write(f"{spk} {' '.join(utts)}\n")
    return spk2utt


def slice_global_features(out_dir, utts, speakers):
    utt2embedding = torch.load(GLOBAL_DIR / "utt2embedding.pt")
    utt2speech_token = torch.load(GLOBAL_DIR / "utt2speech_token.pt")
    spk2embedding = torch.load(GLOBAL_DIR / "spk2embedding.pt")
    torch.save({u: utt2embedding[u] for u in utts if u in utt2embedding}, out_dir / "utt2embedding.pt")
    torch.save({u: utt2speech_token[u] for u in utts if u in utt2speech_token}, out_dir / "utt2speech_token.pt")
    torch.save({s: spk2embedding[s] for s in speakers if s in spk2embedding}, out_dir / "spk2embedding.pt")


def build_regime(name, target_langs):
    # Loaded once per regime and reused for both splits -- these two files are global
    # (utt_id-keyed across all splits combined), not split-specific, so there's no reason to
    # reload them a second time for dev.
    embedding_keys = set(torch.load(GLOBAL_DIR / "utt2embedding.pt").keys())
    token_keys = set(torch.load(GLOBAL_DIR / "utt2speech_token.pt").keys())
    for split in ("train", "dev"):
        global_utt2spk = _global_utt2spk_cache.setdefault(split, _load_global_utt2spk(split))
        valid_utts = set(global_utt2spk.keys()) & embedding_keys & token_keys
        utt2wav, utt2text, utt2spk = collect_utts(target_langs, split, valid_utts)
        out_dir = OUT_ROOT / name / split
        spk2utt = write_kaldi_dir(out_dir, utt2wav, utt2text, utt2spk)
        slice_global_features(out_dir, list(utt2wav.keys()), list(spk2utt.keys()))
        print(f"[{name}/{split}] {len(utt2wav)} utterances, {len(spk2utt)} speakers -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", required=True, help="cluster name, or a single target_lang code")
    parser.add_argument("--kind", required=True, choices=["cluster", "individual"])
    args = parser.parse_args()

    if args.kind == "cluster":
        target_langs = CLUSTERS[args.regime]
        build_regime(f"cluster_{args.regime}", target_langs)
    else:
        build_regime(f"individual_{args.regime}", [args.regime])


if __name__ == "__main__":
    main()
