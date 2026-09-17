"""
Scans a shard of Hausa utterances with PANNs (AudioSet-pretrained CNN14) looking for real,
identifiable background events -- car horn, rain, children playing, crosstalk/crowd, cooking
sounds -- alongside speech. Unlike the denoising job's loudness-only noise-floor metric, this
actually classifies clip content, so it can tell "loud generic hiss" apart from "a horn is
audible in this recording."

Writes one JSON line per utterance whose top-10 AudioSet tags include any target class above
CONF_THRESHOLD, to a per-shard hits file -- non-matches aren't written at all, so hits files
stay small regardless of shard size.
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from panns_inference import AudioTagging, labels

KALDI_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data"
HITS_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/noisy_event_hits")

TARGET_KEYWORDS = {
    "horn": ["horn", "honking"],
    "rain": ["rain"],
    "children_playing": ["children playing", "child speech", "children shouting"],
    "crosstalk": ["conversation", "crowd", "hubbub", "babble"],
    "cooking": ["chopping (food)", "frying (food)", "sizzle"],
    "traffic": ["traffic noise", "motor vehicle", "vehicle "],
}
ALL_KEYWORDS = [kw for kws in TARGET_KEYWORDS.values() for kw in kws]
CONF_THRESHOLD = 0.05


def category_for_label(label_lower):
    for cat, kws in TARGET_KEYWORDS.items():
        if any(kw in label_lower for kw in kws):
            return cat
    return None


def load_hausa_utts():
    kaldi_dir = f"{KALDI_ROOT}/individual_ha-NG"
    wavscp = {}
    with open(f"{kaldi_dir}/dev/wav.scp") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wavscp[parts[0]] = parts[1]
    text = {}
    with open(f"{kaldi_dir}/dev/text") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                text[parts[0]] = parts[1]
    # ha-NG's own kaldi dir only has dev-sized data; use the GLOBAL train wav.scp filtered to
    # hausa for real scale, falling back to dev text for any transcript we can find
    global_train = f"{KALDI_ROOT}/global/train/wav.scp"
    with open(global_train) as f:
        for line in f:
            uid, path = line.strip().split(maxsplit=1)
            if "/speech_out/hausa/" in path and uid not in wavscp:
                wavscp[uid] = path
    return wavscp, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-idx", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    HITS_DIR.mkdir(parents=True, exist_ok=True)
    hits_out = HITS_DIR / f"shard_{args.shard_idx:05d}.jsonl"
    done_marker = HITS_DIR / f"shard_{args.shard_idx:05d}.done"
    if done_marker.exists():
        print(f"shard {args.shard_idx} already done, skipping")
        return

    wavscp, text = load_hausa_utts()
    all_utts = sorted(wavscp.keys())
    shard_utts = all_utts[args.shard_idx :: args.num_shards]
    print(f"shard {args.shard_idx}/{args.num_shards}: {len(shard_utts)} utterances", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioTagging(checkpoint_path=None, device=device)
    print(f"model loaded on {device}", flush=True)

    n_hits = 0
    with open(hits_out, "w") as hf:
        for i, utt in enumerate(shard_utts):
            path = wavscp[utt]
            try:
                audio, sr = librosa.load(path, sr=32000, mono=True)
                if len(audio) < 3200:  # skip near-empty clips
                    continue
                clipwise = model.inference(audio[None, :])[0][0]
            except Exception as e:
                continue

            top_idx = np.argsort(clipwise)[::-1][:10]
            matches = []
            for idx in top_idx:
                conf = float(clipwise[idx])
                if conf < CONF_THRESHOLD:
                    continue
                label_lower = labels[idx].lower()
                cat = category_for_label(label_lower)
                if cat is not None:
                    matches.append({"category": cat, "label": labels[idx], "confidence": round(conf, 4)})

            if matches:
                n_hits += 1
                hf.write(json.dumps({
                    "utt": utt, "path": path, "transcript": text.get(utt, ""),
                    "matches": matches,
                }, ensure_ascii=False) + "\n")
                hf.flush()

            if (i + 1) % 1000 == 0:
                print(f"  [{args.shard_idx}] {i + 1}/{len(shard_utts)} processed, {n_hits} hits so far", flush=True)

    print(f"shard {args.shard_idx} done: {n_hits} hits out of {len(shard_utts)}", flush=True)
    done_marker.touch()


if __name__ == "__main__":
    main()
