"""Did training stop because a stage converged, or because it hit the step cap?

A stage ends when EITHER its step target is reached OR its CV loss stops improving for three
consecutive evals. Those endings mean opposite things:

  plateau -> the model had stopped learning; more steps would not have helped
  cap     -> the clock ran out while CV loss was still falling; it was cut short

Caps are 60,000 steps for an individual llm or flow, and 150,000 for the combined regime.

For each regime this reads the CV curve and reports where it ended, whether the minimum was
at the very end (still improving) or well before it (diverged, so stopping was right), and
how much loss was still coming off per 10k steps at the point it stopped.
"""
import os
import re
import sys
from collections import defaultdict

R = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints"
CAPS = {"individual": 60000, "cluster": 100000, "combined": 150000}


def curve(tb_dir, stage):
    """(step, cv_loss) pairs for one stage, from its tensorboard events."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        print("tensorboard not available:", e); sys.exit(1)
    d = os.path.join(tb_dir, stage)
    if not os.path.isdir(d):
        return []
    pts = []
    for fn in sorted(os.listdir(d)):
        if "tfevents" not in fn:
            continue
        try:
            ea = EventAccumulator(os.path.join(d, fn), size_guidance={"scalars": 0})
            ea.Reload()
            tags = [t for t in ea.Tags().get("scalars", []) if "cv" in t.lower() and "loss" in t.lower()]
            if not tags:
                tags = [t for t in ea.Tags().get("scalars", []) if t.lower().endswith("loss")]
            for t in tags[:1]:
                pts += [(e.step, e.value) for e in ea.Scalars(t)]
        except Exception:
            continue
    return sorted(set(pts))


def main():
    kinds = [("individual", f"{R}/individual"), ("combined", f"{R}/combined")]
    print(f"{'regime':26}{'stage':6}{'end step':>10}{'cap':>8}{'min cv':>9}"
          f"{'final cv':>10}{'min at':>9}   verdict")
    print("-" * 104)
    rows = []
    for kind, base in kinds:
        if not os.path.isdir(base):
            continue
        for reg in sorted(os.listdir(base)):
            tb = os.path.join(base, reg, "tensorboard")
            if not os.path.isdir(tb):
                continue
            for stage in ("llm", "flow"):
                pts = curve(tb, stage)
                if len(pts) < 3:
                    continue
                steps = [p[0] for p in pts]
                vals = [p[1] for p in pts]
                end = max(steps)
                mn = min(vals)
                mn_step = steps[vals.index(mn)]
                final = vals[-1]
                cap = CAPS[kind]
                # still improving = the best CV is at (or very near) the last eval
                still = mn_step >= end * 0.9
                hit_cap = end >= cap * 0.95
                if still and hit_cap:
                    v = "CUT SHORT -- still improving at the cap"
                elif still:
                    v = "stopped early, still improving"
                elif final > mn * 1.5:
                    v = f"diverged ({final/mn:.1f}x off its min)"
                else:
                    v = "converged"
                # how fast was it still learning at the end?
                tail = [(s, x) for s, x in pts if s >= end * 0.7]
                slope = ((tail[0][1] - tail[-1][1]) / max(1, (tail[-1][0] - tail[0][0])) * 10000
                         if len(tail) > 1 else 0.0)
                rows.append((reg, stage, end, cap, mn, final, mn_step, v, slope))
                print(f"{reg[:25]:26}{stage:6}{end:>10}{cap:>8}{mn:>9.4f}{final:>10.4f}"
                      f"{mn_step:>9}   {v}")
    cut = [r for r in rows if "CUT SHORT" in r[7]]
    still = [r for r in rows if r[7] == "stopped early, still improving"]
    div = [r for r in rows if "diverged" in r[7]]
    print(f"\n=== {len(cut)} stage(s) cut short at the cap while still improving ===")
    for r in sorted(cut, key=lambda x: -x[8]):
        print(f"   {r[0]:24} {r[1]:5} ended {r[2]}, still shedding {r[8]:.4f} cv/10k steps")
    print(f"\n=== {len(still)} stopped early while still improving (patience fired on noise?) ===")
    for r in sorted(still, key=lambda x: -x[8])[:12]:
        print(f"   {r[0]:24} {r[1]:5} ended {r[2]} of {r[3]}, {r[8]:.4f} cv/10k")
    print(f"\n=== {len(div)} diverged (stopping was right) ===")
    for r in div[:10]:
        print(f"   {r[0]:24} {r[1]:5} {r[7]}")


if __name__ == "__main__":
    main()
