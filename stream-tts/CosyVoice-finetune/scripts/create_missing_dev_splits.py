"""
One-time fix: bm-ML, fon-BJ, kri-SL, kr-NG, ss-SZ have no dev-split manifest at all (missing
even in the raw nemotron_ft/manifests/ source, not just manifests_clean/ -- confirmed upstream
data-prep gap, not a bug in 04_build_regime_kaldi.py). Their empty dev/wav.scp made the CosyVoice
llm training's CV DataLoader crash with ZeroDivisionError on every attempt (empty per-worker
shard list), which the pipeline's own auto-resubmit then retried in an infinite crash loop --
see conversation.

Writes manifests_clean/{lang}_dev.jsonl as a deterministic (seed=42) random sample of each
language's existing manifests_clean/{lang}_train.jsonl -- train.jsonl itself is left completely
untouched (not rewritten to remove the sampled rows) since that file is shared with the ASR
(Nemotron) pipeline and this fix's only goal is unblocking CosyVoice's CV evaluation for these
5 regimes, not strict train/dev hygiene. A small amount of train/dev overlap is an accepted,
deliberate tradeoff over risking any side effect on ASR training data.

Dev size: min(3000, max(50, round(0.05 * train_count))) -- roughly matches the scale of
existing languages' dev splits (e.g. ha-NG: 1.88M train / 3158 dev) without taking an outsized
fraction of these much smaller train sets.
"""
import json
import random

MANIFEST_DIR = "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/manifests_clean"
LANGS = ["bm-ML", "fon-BJ", "kri-SL", "kr-NG", "ss-SZ"]
SEED = 42


def main():
    for lang in LANGS:
        train_path = f"{MANIFEST_DIR}/{lang}_train.jsonl"
        dev_path = f"{MANIFEST_DIR}/{lang}_dev.jsonl"

        with open(train_path) as f:
            rows = f.readlines()
        n_train = len(rows)
        n_dev = min(3000, max(50, round(0.05 * n_train)))

        rng = random.Random(SEED)
        dev_rows = rng.sample(rows, n_dev)

        with open(dev_path, "w") as f:
            f.writelines(dev_rows)

        print(f"{lang}: train={n_train} (untouched) -> wrote dev={n_dev} to {dev_path}")


if __name__ == "__main__":
    main()
