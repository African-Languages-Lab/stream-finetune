"""
Pushes each Nemotron ASR and CosyVoice3 TTS regime's *latest* checkpoint (not the full saved
history) to its own model repo under the all-lab HF namespace (HF_TOKEN_ORGOWNER's own user
account -- huggingface.co/all-lab; also admin on the African-Languages-Lab org, but the latter
turned out not to be what "all-lab" refers to).

Latest-only, not full history: CosyVoice keeps every training step's checkpoint uncapped (e.g.
individual_en-UG alone had 198 step checkpoints, 374GB, for one regime's first of three stages),
so pushing everything saved would have meant ~5.4TB and climbing, almost all of it redundant
snapshots of the same in-progress training run. Only the current best-known state per regime/
stage is actually useful for using or resuming from the model.

Each regime/stage now uploads at most two small files (the checkpoint + its sibling .yaml config
for CosyVoice) via upload_file, not upload_large_folder -- this also sidesteps the account's hard
600s CPU-time cap (confirmed to apply everywhere: interactive shell, nohup, and `at` jobs alike,
while SLURM compute nodes have no internet at all): a single ~2-8GB file uploads well within that
budget, unlike the previous full-folder uploads for large regimes which routinely got killed
mid-transfer. Regime-level `_hf_done` markers still let push_runner.sh's self-rescheduling `at`
chain skip already-pushed regimes on resubmit.
"""
import argparse
import re
import time
from pathlib import Path

from huggingface_hub import HfApi

ORG = "all-lab"

NEMOTRON_COLLECTION_TITLE = "Nemotron ASR — African Language Fine-tunes"
NEMOTRON_COLLECTION_DESC = (
    "Nemotron 3.5 ASR (0.6B) fine-tuned per-language/cluster/combined on 41 African "
    "language codes. Snapshot of active training, not a finished release."
)

COSYVOICE_COLLECTION_TITLE = "CosyVoice3 TTS — African Language Fine-tunes"
COSYVOICE_COLLECTION_DESC = (
    "CosyVoice3-0.5B (llm/flow/hifigan) fine-tuned per-language/cluster/combined on "
    "41 African language codes. Snapshot of active training, not finished."
)


def get_or_create_collections(api: HfApi):
    nemo_coll = api.create_collection(
        title=NEMOTRON_COLLECTION_TITLE, namespace=ORG, description=NEMOTRON_COLLECTION_DESC,
        private=True, exists_ok=True,
    )
    cv_coll = api.create_collection(
        title=COSYVOICE_COLLECTION_TITLE, namespace=ORG, description=COSYVOICE_COLLECTION_DESC,
        private=True, exists_ok=True,
    )
    return nemo_coll.slug, cv_coll.slug


NEMOTRON_ROOT = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints")
COSYVOICE_ROOT = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints")
DONE_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/hf_push_done")
DONE_DIR.mkdir(parents=True, exist_ok=True)

NEMOTRON_CLUSTERS = ["afroasiatic", "east_africa_bantu", "southern_central_bantu",
                     "west_africa_niger_congo", "other_languages"]
NEMOTRON_LANGS = "rw-RW ar-AR sw-KE en-GH ha-NG ig-NG st-ZA yo-NG ts-ZA tn-BW zu-ZA nd-ZW bm-ML ff-SN ki-KE ny-MW xh-ZA so-SO tw-GH ve-ZA am-ET lg-UG ti-ER or-KE ss-SZ af-ZA bem-ZM nso-ZA ee-GH mg-MG ln-CD sn-ZW kr-NG fon-BJ ber-MA wo-SN kri-SL en-ZA umb-AO en-UG en-NG".split()

COSYVOICE_CLUSTERS = NEMOTRON_CLUSTERS
COSYVOICE_LANGS = "af-ZA kr-NG mg-MG kri-SL en-GH en-NG en-ZA en-UG ar-AR am-ET ti-ER ha-NG or-KE so-SO ber-MA ki-KE rw-RW lg-UG sw-KE ny-MW ln-CD nd-ZW nso-ZA st-ZA sn-ZW ss-SZ ts-ZA tn-BW umb-AO ve-ZA xh-ZA zu-ZA bem-ZM bm-ML ee-GH fon-BJ ff-SN ig-NG tw-GH wo-SN yo-NG".split()
COSYVOICE_STAGES = ["llm", "flow", "hifigan"]


def nemotron_checkpoint_dir(kind: str, name: str) -> Path:
    if kind == "individual":
        return NEMOTRON_ROOT / "individual" / name / f"individual_{name}" / "checkpoints"
    if kind == "cluster":
        return NEMOTRON_ROOT / "clusters" / name / f"cluster_{name}" / "checkpoints"
    return NEMOTRON_ROOT / "combined" / "nemotron_combined_multilingual" / "checkpoints"


def latest_nemotron_checkpoint(ckpt_dir: Path):
    """Prefer a final exported .nemo; else the highest-epoch non-unfinished .ckpt.

    Returns None if nothing usable exists (no data ever trained, or the only checkpoint is
    marked unfinished by NeMo itself -- confirmed real case for ts-ZA, whose own resume logic
    refuses to load such a file, so it isn't a real "available" checkpoint).
    """
    if not ckpt_dir.is_dir():
        return None
    nemo_files = sorted(ckpt_dir.glob("*.nemo"))
    if nemo_files:
        return nemo_files[-1]
    unfinished_stems = {p.name[: -len("-unfinished")] for p in ckpt_dir.glob("*-unfinished")}
    candidates = [p for p in ckpt_dir.glob("*.ckpt") if p.name not in unfinished_stems]
    if not candidates:
        return None

    def step_key(p: Path):
        m = re.search(r"epoch=(\d+)", p.name)
        return int(m.group(1)) if m else -1

    return sorted(candidates, key=step_key)[-1]


def cosyvoice_stage_dir(kind: str, name: str, stage: str) -> Path:
    if kind == "individual":
        return COSYVOICE_ROOT / "individual" / f"individual_{name}" / stage
    if kind == "cluster":
        return COSYVOICE_ROOT / "cluster" / f"cluster_{name}" / stage
    return COSYVOICE_ROOT / "combined" / "combined" / stage


def latest_cosyvoice_checkpoint(stage_dir: Path):
    """Highest-step epoch_X_step_Y.pt (plus its sibling .yaml if present), else init.pt."""
    if not stage_dir.is_dir():
        return None
    step_files = list(stage_dir.glob("epoch_*_step_*.pt"))
    if step_files:
        def step_key(p: Path):
            m = re.search(r"step_(\d+)\.pt$", p.name)
            return int(m.group(1)) if m else -1
        pt = sorted(step_files, key=step_key)[-1]
    else:
        pt = stage_dir / "init.pt"
        if not pt.exists():
            return None
    yaml_sibling = pt.with_suffix(".yaml")
    return [pt, yaml_sibling] if yaml_sibling.exists() else [pt]


COOLDOWN_SECONDS = 15 * 60  # matches roughly how long HF's per-repo commit rate limit takes to clear


def push_files(api: HfApi, repo_id: str, files, done_marker: Path, collection_slug: str):
    if done_marker.exists():
        print(f"[skip] {repo_id} already pushed")
        return
    if not files:
        print(f"[skip] {repo_id}: no checkpoint files on disk yet")
        return
    fail_marker = DONE_DIR / f"{repo_id.replace('/', '__')}.failed"
    if fail_marker.exists() and time.time() - fail_marker.stat().st_mtime < COOLDOWN_SECONDS:
        # Retrying a rate-limited repo every single pass (every ~1 minute via the `at` reschedule
        # chain) kept re-extending its own lockout window instead of waiting it out (confirmed:
        # nemotron-asr-cluster-southern_central_bantu's "retry in N minutes" grew from 33 to 20
        # across repeated 1-minute-apart attempts). Skip it until the cooldown clears.
        print(f"[skip] {repo_id}: cooling down after a recent failure")
        return
    try:
        _push_files_inner(api, repo_id, files, done_marker, collection_slug)
        fail_marker.unlink(missing_ok=True)
    except Exception as e:
        fail_marker.parent.mkdir(parents=True, exist_ok=True)
        fail_marker.touch()
        # A single repo hitting HF's per-repo commit rate limit (confirmed real: nemotron-asr-
        # cluster-southern_central_bantu) must not take down the whole pass -- an unhandled
        # exception here previously meant every other not-yet-attempted regime in this run was
        # silently skipped too, and each 1-minute reschedule re-hit the SAME blocked repo,
        # extending its own lockout window instead of waiting it out. Log and move on; the
        # blocked repo gets retried on the next scheduled pass along with everything else.
        print(f"[error] {repo_id} failed, skipping for this pass: {e}")


def _push_files_inner(api: HfApi, repo_id: str, files, done_marker: Path, collection_slug: str):
    api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
    for f in files:
        print(f"[upload] {repo_id} <- {f}")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            commit_message=f"Add {f.name}",
        )
    api.add_collection_item(collection_slug, item_id=repo_id, item_type="model", exists_ok=True)
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.touch()
    print(f"[done] {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["nemotron", "cosyvoice"], default=None)
    args = parser.parse_args()

    api = HfApi()
    nemo_slug, cv_slug = get_or_create_collections(api)

    nemotron_regimes = [("combined", "combined")]
    nemotron_regimes += [("cluster", c) for c in NEMOTRON_CLUSTERS]
    nemotron_regimes += [("individual", l) for l in NEMOTRON_LANGS]

    cosyvoice_regimes = [("combined", "combined")]
    cosyvoice_regimes += [("cluster", c) for c in COSYVOICE_CLUSTERS]
    cosyvoice_regimes += [("individual", l) for l in COSYVOICE_LANGS]

    if args.only in (None, "nemotron"):
        for kind, name in nemotron_regimes:
            repo_id = f"{ORG}/nemotron-asr-{kind}" if kind == "combined" else f"{ORG}/nemotron-asr-{kind}-{name}"
            done_marker = DONE_DIR / f"{repo_id.replace('/', '__')}.done"
            ckpt = latest_nemotron_checkpoint(nemotron_checkpoint_dir(kind, name))
            push_files(api, repo_id, [ckpt] if ckpt else [], done_marker, nemo_slug)

    if args.only in (None, "cosyvoice"):
        for kind, name in cosyvoice_regimes:
            for stage in COSYVOICE_STAGES:
                repo_id = (f"{ORG}/cosyvoice3-{stage}-{kind}" if kind == "combined"
                           else f"{ORG}/cosyvoice3-{stage}-{kind}-{name}")
                done_marker = DONE_DIR / f"{repo_id.replace('/', '__')}.done"
                files = latest_cosyvoice_checkpoint(cosyvoice_stage_dir(kind, name, stage))
                push_files(api, repo_id, files or [], done_marker, cv_slug)

    print("=== push run complete ===")


if __name__ == "__main__":
    main()
