"""
Merges all per-shard utt2embedding.pt / utt2speech_token.pt files (from
02_extract_features_shard.py) into two global dicts, and computes spk2embedding by
averaging utt2embedding per pseudo-speaker (from the global utt2spk mapping). This can't be
done per-shard: a pseudo-speaker's utterances are scattered across many shards (226 speakers
over ~12.9M utterances), so a per-shard average would be wrong.
"""
import json
from pathlib import Path

import numpy as np
import torch

FEATURES_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/features")
GLOBAL_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data/global")

# Shards store embeddings/tokens as plain Python lists (from .tolist() in the extraction
# script). At ~12.9M utterances, a dict of Python-list-of-float objects is enormous --
# per-object overhead alone (each Python float ~24-32 bytes vs 4 bytes packed) pushed a
# 200GB merge job to OOM. Converting to compact numpy arrays here cuts memory ~6-8x; still
# converts cleanly back to a tensor wherever downstream code consumes it.


def load_utt2spk():
    utt2spk = {}
    for split in ("train", "dev"):
        with open(GLOBAL_DIR / split / "utt2spk") as f:
            for line in f:
                utt, spk = line.rstrip("\n").split(" ", 1)
                utt2spk[utt] = spk
    return utt2spk


def main():
    shard_embs = sorted(FEATURES_DIR.glob("shard_*.utt2embedding.pt"))
    shard_toks = sorted(FEATURES_DIR.glob("shard_*.utt2speech_token.pt"))
    done_markers = sorted(FEATURES_DIR.glob("shard_*.done"))
    print(f"found {len(shard_embs)} embedding shards, {len(shard_toks)} token shards, "
          f"{len(done_markers)} done markers")
    if len(shard_embs) != len(shard_toks) or len(shard_embs) != len(done_markers):
        raise RuntimeError("shard counts don't match -- extraction is incomplete, not merging")

    utt2embedding = {}
    for p in shard_embs:
        d = torch.load(p)
        for utt, emb in d.items():
            utt2embedding[utt] = np.asarray(emb, dtype=np.float32)
        del d
    print(f"merged utt2embedding: {len(utt2embedding)} utterances")

    utt2speech_token = {}
    n_empty = 0
    for p in shard_toks:
        d = torch.load(p)
        for utt, tok in d.items():
            if len(tok) == 0:
                n_empty += 1
                continue  # dropped: tokenizer refused (e.g. audio >30s that slipped through)
            utt2speech_token[utt] = np.asarray(tok, dtype=np.int32)
        del d
    print(f"merged utt2speech_token: {len(utt2speech_token)} utterances ({n_empty} empty/dropped)")

    # utt2embedding/utt2speech_token were computed against the original (pre-cleaning) global
    # utterance set; utt2spk was rebuilt afterward from the cleaned manifests, so utt_ids
    # belonging to corrupted rows dropped during cleaning legitimately won't appear here --
    # those utterances are excluded from every regime anyway, so drop them here too rather
    # than carry stale entries into the saved global feature files.
    utt2spk = load_utt2spk()
    n_before = len(utt2embedding)
    utt2embedding = {u: e for u, e in utt2embedding.items() if u in utt2spk}
    utt2speech_token = {u: t for u, t in utt2speech_token.items() if u in utt2spk}
    print(f"dropped {n_before - len(utt2embedding)} embeddings belonging to cleaned-out utterances")

    spk2embs = {}
    for utt, emb in utt2embedding.items():
        spk2embs.setdefault(utt2spk[utt], []).append(emb)
    spk2embedding = {spk: np.mean(np.stack(embs), axis=0).astype(np.float32) for spk, embs in spk2embs.items()}
    print(f"computed spk2embedding for {len(spk2embedding)} speakers")

    torch.save(utt2embedding, GLOBAL_DIR / "utt2embedding.pt")
    torch.save(spk2embedding, GLOBAL_DIR / "spk2embedding.pt")
    torch.save(utt2speech_token, GLOBAL_DIR / "utt2speech_token.pt")

    summary = {
        "n_utt2embedding": len(utt2embedding),
        "n_utt2speech_token": len(utt2speech_token),
        "n_speakers": len(spk2embedding),
        "n_empty_tokens_dropped": n_empty,
    }
    with open(GLOBAL_DIR / "merge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
