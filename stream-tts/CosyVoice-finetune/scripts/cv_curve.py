"""Print the CV-loss trajectory for named regimes, so "is it still improving?" is answerable.

A stage ends on whichever comes first: its step cap, or CV loss failing to improve for three
consecutive evals. Those mean opposite things -- a plateau says more steps would not have
helped, a cap says the clock ran out while the model was still learning. The only way to tell
them apart is to look at where the minimum sits relative to the end of the run.
"""
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

R = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints"
TARGETS = [("hausa", f"{R}/individual/individual_ha-NG", 60000),
           ("igbo", f"{R}/individual/individual_ig-NG", 60000),
           ("combined", f"{R}/combined/combined", 150000)]


def curve(tb, stage):
    d = os.path.join(tb, stage)
    if not os.path.isdir(d):
        return []
    pts = []
    for fn in sorted(os.listdir(d)):
        if "tfevents" not in fn:
            continue
        try:
            ea = EventAccumulator(os.path.join(d, fn), size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            pick = [t for t in tags if "cv" in t.lower() and "loss" in t.lower()] or \
                   [t for t in tags if t.lower().endswith("loss")]
            for t in pick[:1]:
                pts += [(e.step, e.value) for e in ea.Scalars(t)]
        except Exception:
            continue
    return sorted(set(pts))


for name, base, cap in TARGETS:
    tb = os.path.join(base, "tensorboard")
    print(f"\n{'='*70}\n{name}   (step cap was {cap})\n{'='*70}")
    for stage in ("llm", "flow"):
        pts = curve(tb, stage)
        if len(pts) < 2:
            print(f"  {stage}: no CV curve found")
            continue
        steps = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        end, mn = max(steps), min(vals)
        mn_step = steps[vals.index(mn)]
        # sample the curve so the shape is visible without printing hundreds of points
        n = len(pts)
        idx = sorted({0, n // 6, n // 3, n // 2, (2 * n) // 3, (5 * n) // 6, n - 1})
        print(f"\n  --- {stage}: {n} evals, ends at step {end}")
        print(f"      {'step':>9}  {'cv loss':>9}")
        for i in idx:
            s, v = pts[i]
            bar = "#" * max(1, int((v - mn) / max(1e-9, max(vals) - mn) * 34)) if max(vals) > mn else ""
            print(f"      {s:>9}  {v:>9.4f}  {bar}")
        tail = [p for p in pts if p[0] >= end * 0.7]
        slope = ((tail[0][1] - tail[-1][1]) / max(1, tail[-1][0] - tail[0][0]) * 10000) if len(tail) > 1 else 0.0
        still = mn_step >= end * 0.9
        print(f"      min {mn:.4f} at step {mn_step}   final {vals[-1]:.4f}")
        print(f"      last 30% of the run: {slope:+.4f} cv per 10k steps "
              f"({'still falling' if slope > 0.001 else 'flat or rising'})")
        print(f"      VERDICT: {'STILL IMPROVING when it stopped' if still else 'had already bottomed out'}"
              + (f" -- and it stopped AT the {cap} cap" if end >= cap * 0.95 else ""))
