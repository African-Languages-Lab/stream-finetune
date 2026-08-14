"""
One-off smoke test setup: builds a tiny Kaldi dir from the already-completed extraction
smoke test (2571 utterances, archived under features/_smoke_test_archive/) so the training
side of the pipeline (parquet packaging, llm/flow/hifigan torchrun stages, checkpoint
resume, the hift.pt naming fix) can be validated end-to-end without waiting for the full
12.87M-utterance extraction array to finish. Not part of the real pipeline -- writes to a
"smoke" regime dir that submit_all_regimes.sh never references.
"""
from pathlib import Path

import torch

GLOBAL_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data/global")
ARCHIVE = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/features/_smoke_test_archive")
OUT = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data/smoke")


def load_scp(path):
    d = {}
    with open(path) as f:
        for line in f:
            k, v = line.rstrip("\n").split(" ", 1)
            d[k] = v
    return d


def main():
    utt2embedding = torch.load(ARCHIVE / "shard_00000.utt2embedding.pt")
    utt2speech_token = torch.load(ARCHIVE / "shard_00000.utt2speech_token.pt")
    utt2speech_token = {k: v for k, v in utt2speech_token.items() if len(v) > 0}
    smoke_utts = set(utt2embedding.keys()) & set(utt2speech_token.keys())
    print(f"smoke utterance pool: {len(smoke_utts)}")

    # the extraction shard was sampled from train+dev combined, so an utt_id here may
    # originate from either split's original manifest -- merge both before slicing
    global_wav, global_text, global_spk = {}, {}, {}
    for split in ("train", "dev"):
        global_wav.update(load_scp(GLOBAL_DIR / split / "wav.scp"))
        global_text.update(load_scp(GLOBAL_DIR / split / "text"))
        global_spk.update(load_scp(GLOBAL_DIR / split / "utt2spk"))

    # 90/10 split for a train/dev smoke pair
    smoke_utts = sorted(smoke_utts)
    n_dev = max(1, len(smoke_utts) // 10)
    splits = {"dev": smoke_utts[:n_dev], "train": smoke_utts[n_dev:]}

    for split, utts in splits.items():
        out_dir = OUT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        spk2utt = {}
        with open(out_dir / "wav.scp", "w") as fw, open(out_dir / "text", "w") as ft, \
                open(out_dir / "utt2spk", "w") as fs:
            for u in utts:
                if u not in global_wav:
                    continue  # not a train-split utt (dev-split utts live in a different scp)
                fw.write(f"{u} {global_wav[u]}\n")
                ft.write(f"{u} {global_text[u]}\n")
                spk = global_spk[u]
                fs.write(f"{u} {spk}\n")
                spk2utt.setdefault(spk, []).append(u)
        with open(out_dir / "spk2utt", "w") as f:
            for spk, us in spk2utt.items():
                f.write(f"{spk} {' '.join(us)}\n")
        kept = list(spk2utt.values())
        n_kept = sum(len(v) for v in kept)
        torch.save({u: utt2embedding[u] for u in utts if u in utt2embedding and u in global_wav},
                   out_dir / "utt2embedding.pt")
        torch.save({u: utt2speech_token[u] for u in utts if u in utt2speech_token and u in global_wav},
                   out_dir / "utt2speech_token.pt")
        spk2emb = {}
        for spk, us in spk2utt.items():
            embs = [utt2embedding[u] for u in us if u in utt2embedding]
            if embs:
                spk2emb[spk] = torch.tensor(embs).mean(dim=0).tolist()
        torch.save(spk2emb, out_dir / "spk2embedding.pt")
        print(f"[{split}] {n_kept} utterances, {len(spk2utt)} speakers -> {out_dir}")


if __name__ == "__main__":
    main()
