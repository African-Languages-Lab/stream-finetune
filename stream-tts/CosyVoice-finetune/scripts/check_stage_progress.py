"""
Decides whether a regime's current training stage should advance (target step count reached,
OR CV loss has stopped improving for PATIENCE consecutive evals) or continue. Called by
run_regime_train.sbatch after each SLURM job segment ends (wall-time cutoff or natural exit).

Early stopping exists because a full audit (audit_llm_divergence.py) found 30 of 33 individual
llm regimes' CV loss diverges (rises 2.5-4.6x from its minimum) well before any of the new
step targets -- training a fixed number of steps regardless of CV trend would just reproduce
that on the new, larger step budgets. Patience is counted in evals, not steps, since
save_per_step differs by stage (10k llm/flow, 25k hifigan) -- eval cadence follows the same
interval, so a fixed eval-count patience means noise tolerance ends up naturally.
"""
import argparse
import glob
import json
import os
import sys

STAGES = ["llm", "flow", "hifigan"]

STEP_TARGETS = {
    "individual": {"llm": 60000, "flow": 60000, "hifigan": 250000},
    "cluster": {"llm": 100000, "flow": 100000, "hifigan": 350000},
    "combined": {"llm": 150000, "flow": 150000, "hifigan": 500000},
}

PATIENCE = 3  # consecutive non-improving CV evals before declaring this stage converged


def current_step(model_dir):
    pts = [p for p in glob.glob(f"{model_dir}/*.pt") if os.path.basename(p) != "init.pt"]
    if not pts:
        return 0
    import torch
    latest = max(pts, key=os.path.getmtime)
    try:
        sd = torch.load(latest, map_location="cpu", weights_only=True)
        return sd.get("step", 0) if isinstance(sd, dict) else 0
    except Exception:
        return 0


def latest_cv_loss(tb_dir):
    from tensorboard.backend.event_processing import event_accumulator
    files = sorted(glob.glob(f"{tb_dir}/events.out.tfevents.*"))
    if not files:
        return None, None
    best_tag_points = {}
    for fpath in files:
        try:
            ea = event_accumulator.EventAccumulator(fpath, size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags().get("scalars", []):
            if tag != "CV/loss":
                continue
            for ev in ea.Scalars(tag):
                best_tag_points[ev.step] = ev.value
    if not best_tag_points:
        return None, None
    last_step = max(best_tag_points.keys())
    return last_step, best_tag_points[last_step]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--kind", required=True, choices=STEP_TARGETS.keys())
    ap.add_argument("--state-file", required=True)
    args = ap.parse_args()

    state = {"stage_idx": 0, "best_loss": None, "patience_used": 0}
    if os.path.exists(args.state_file):
        state.update(json.load(open(args.state_file)))

    stage_idx = state["stage_idx"]
    model = STAGES[stage_idx]
    model_dir = f"{args.checkpoint_dir}/{model}"
    tb_dir = f"{args.checkpoint_dir}/tensorboard/{model}"

    step = current_step(model_dir)
    target = STEP_TARGETS[args.kind][model]
    # hifigan's GAN loss is multi-component (loss_gen/loss_fm/loss_mel/loss_tpr/loss_f0 summed)
    # and adversarial training is expected to oscillate rather than monotonically improve the
    # way llm's plain cross-entropy did -- the divergence pattern that motivated CV early
    # stopping (audit_llm_divergence.py) was only ever established for llm, never for hifigan.
    # Applying the same patience logic here risked reading normal GAN noise as "stopped
    # improving" and cutting training short before the from-scratch discriminator has had
    # anywhere near enough steps to converge. hifigan always trains to its full step target,
    # no early stop, regardless of kind (individual/cluster/combined).
    eval_step, cv_loss = latest_cv_loss(tb_dir) if model != "hifigan" else (None, None)

    advance = False
    reason = ""
    if step >= target:
        advance = True
        reason = f"step target reached ({step} >= {target})"
    elif cv_loss is not None:
        best_loss = state["best_loss"]
        if best_loss is None or cv_loss < best_loss - 1e-4:
            state["best_loss"] = cv_loss
            state["patience_used"] = 0
        else:
            state["patience_used"] += 1
        if state["patience_used"] >= PATIENCE:
            advance = True
            reason = (f"CV loss stopped improving ({PATIENCE} evals with no improvement, "
                       f"best={state['best_loss']:.4f} at step {eval_step})")

    print(f"[{model}] step={step}/{target} cv_loss={cv_loss} best={state['best_loss']} "
          f"patience={state['patience_used']}/{PATIENCE}", file=sys.stderr)

    if advance:
        print(f"=== advancing: {reason} ===", file=sys.stderr)
        stage_idx += 1
        state = {"stage_idx": stage_idx, "best_loss": None, "patience_used": 0}

    json.dump(state, open(args.state_file, "w"))
    print(stage_idx)


if __name__ == "__main__":
    main()
