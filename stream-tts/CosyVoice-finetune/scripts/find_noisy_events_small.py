"""
Small, targeted version of find_noisy_events_shard.py: scans Hausa utterances one at a time
and stops as soon as it finds MAX_HITS genuine background-event matches (car horn, rain,
children playing, crosstalk, cooking), instead of committing to scanning a fixed large batch.
"""
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from panns_inference import AudioTagging, labels

KALDI_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data"
OUT_PATH = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/noisy_event_hits/small_hits.jsonl")

TARGET_KEYWORDS = {
    "horn": ["horn", "honking"],
    "rain": ["rain"],
    "children_playing": ["children playing", "child speech", "children shouting"],
    "crosstalk": ["conversation", "crowd", "hubbub", "babble"],
    "cooking": ["chopping (food)", "frying (food)", "sizzle"],
}
CONF_THRESHOLD = 0.05
MAX_HITS = 10
MAX_SCANNED = 3000  # give up after this many if not enough hits found


def category_for_label(label_lower):
    for cat, kws in TARGET_KEYWORDS.items():
        if any(kw in label_lower for kw in kws):
            return cat
    return None


def load_hausa_utts():
    kaldi_dir = f"{KALDI_ROOT}/individual_ha-NG"
    text = {}
    with open(f"{kaldi_dir}/dev/text") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                text[parts[0]] = parts[1]
    wavscp = {}
    with open(f"{KALDI_ROOT}/global/train/wav.scp") as f:
        for line in f:
            uid, path = line.strip().split(maxsplit=1)
            if "/speech_out/hausa/" in path:
                wavscp[uid] = path
    return wavscp, text


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wavscp, text = load_hausa_utts()
    all_utts = sorted(wavscp.keys())
    print(f"scanning up to {MAX_SCANNED} of {len(all_utts)} hausa utterances, stopping at {MAX_HITS} hits", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioTagging(checkpoint_path=None, device=device)
    print(f"model loaded on {device}", flush=True)

    n_hits = 0
    n_scanned = 0
    with open(OUT_PATH, "w") as hf:
        for utt in all_utts:
            if n_hits >= MAX_HITS or n_scanned >= MAX_SCANNED:
                break
            path = wavscp[utt]
            n_scanned += 1
            try:
                audio, sr = librosa.load(path, sr=32000, mono=True)
                if len(audio) < 3200:
                    continue
                clipwise = model.inference(audio[None, :])[0][0]
            except Exception:
                continue

            top_idx = np.argsort(clipwise)[::-1][:10]
            matches = []
            for idx in top_idx:
                conf = float(clipwise[idx])
                if conf < CONF_THRESHOLD:
                    continue
                cat = category_for_label(labels[idx].lower())
                if cat is not None:
                    matches.append({"category": cat, "label": labels[idx], "confidence": round(conf, 4)})

            if matches:
                n_hits += 1
                row = {"utt": utt, "path": path, "transcript": text.get(utt, ""), "matches": matches}
                hf.write(json.dumps(row, ensure_ascii=False) + "\n")
                hf.flush()
                print(f"  HIT {n_hits}/{MAX_HITS} (scanned {n_scanned}): {matches} :: {path}", flush=True)

            if n_scanned % 200 == 0:
                print(f"  scanned {n_scanned}, {n_hits} hits so far", flush=True)

    print(f"\ndone: {n_hits} hits after scanning {n_scanned} utterances", flush=True)


if __name__ == "__main__":
    main()
