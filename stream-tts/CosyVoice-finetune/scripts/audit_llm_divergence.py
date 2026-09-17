"""
Scans EVERY regime's llm tensorboard CV loss/accuracy curve (individual, cluster, combined)
to check whether the divergence pattern found in a 5-language spot-check (am-ET, nso-ZA,
or-KE all diverged catastrophically past ~40-70k steps; rw-RW, sw-KE stayed healthy) is
widespread or isolated. Reports, per regime: initial CV loss/acc, the BEST (lowest-loss)
checkpoint's step, the LATEST checkpoint's step, and whether latest is worse than best
(the divergence signal -- and the reason our bundling script's "pick by mtime" policy may be
silently selecting the worst available checkpoint for some regimes).
"""
import glob
import os
import sys
from tensorboard.backend.event_processing import event_accumulator

CKPT_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints"


def read_cv_curve(tb_dir):
    files = sorted(glob.glob(f"{tb_dir}/events.out.tfevents.*"))
    if not files:
        return None
    steps_loss, steps_acc = {}, {}
    for fpath in files:
        try:
            ea = event_accumulator.EventAccumulator(fpath, size_guidance={'scalars': 0})
            ea.Reload()
        except Exception as e:
            print(f"    (skip unreadable file {os.path.basename(fpath)}: {e})", flush=True)
            continue
        for tag in ea.Tags().get('scalars', []):
            if 'cv' not in tag.lower():
                continue
            if 'loss' in tag.lower():
                target = steps_loss
            elif 'acc' in tag.lower():
                target = steps_acc
            else:
                continue
            for ev in ea.Scalars(tag):
                target[ev.step] = ev.value
    if not steps_loss:
        return None
    return steps_loss, steps_acc


def summarize(lang, tb_dir):
    result = read_cv_curve(tb_dir)
    if result is None:
        print(f"{lang:24s} NO_CV_DATA")
        return
    steps_loss, steps_acc = result
    steps = sorted(steps_loss.keys())
    first_step = steps[0]
    last_step = steps[-1]
    best_step = min(steps, key=lambda s: steps_loss[s])
    first_loss, last_loss, best_loss = steps_loss[first_step], steps_loss[last_step], steps_loss[best_step]
    first_acc = steps_acc.get(first_step)
    last_acc = steps_acc.get(last_step)
    best_acc = steps_acc.get(best_step)
    ratio = last_loss / best_loss if best_loss > 0 else float('inf')
    verdict = "DIVERGED" if ratio > 1.5 else ("drifting" if ratio > 1.15 else "stable")
    print(f"{lang:24s} n_evals={len(steps):4d}  first(step{first_step:>7d})=loss{first_loss:6.3f}/acc{first_acc if first_acc else -1:5.3f}"
          f"  best(step{best_step:>7d})=loss{best_loss:6.3f}/acc{best_acc if best_acc else -1:5.3f}"
          f"  latest(step{last_step:>7d})=loss{last_loss:6.3f}/acc{last_acc if last_acc else -1:5.3f}"
          f"  ratio(latest/best)={ratio:5.2f}x  [{verdict}]", flush=True)


def main():
    print(f"{'regime':24s} summary")
    print("=" * 140)

    print("\n--- INDIVIDUAL ---")
    for path in sorted(glob.glob(f"{CKPT_ROOT}/individual/individual_*")):
        lang = os.path.basename(path)[len("individual_"):]
        tb_dir = f"{path}/tensorboard/llm"
        if os.path.isdir(tb_dir):
            summarize(lang, tb_dir)

    print("\n--- CLUSTER ---")
    for path in sorted(glob.glob(f"{CKPT_ROOT}/cluster/cluster_*")):
        name = os.path.basename(path)
        tb_dir = f"{path}/tensorboard/llm"
        if os.path.isdir(tb_dir):
            summarize(name, tb_dir)

    print("\n--- COMBINED ---")
    tb_dir = f"{CKPT_ROOT}/combined/tensorboard/llm"
    if os.path.isdir(tb_dir):
        summarize("combined", tb_dir)
    else:
        print("combined                 NO_TENSORBOARD_DIR (llm never started, confirmed earlier this session)")


if __name__ == "__main__":
    main()
