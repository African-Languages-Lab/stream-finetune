"""
Stage 0 of the data-prep pipeline (runs before 01_build_global_kaldi.py): for every utterance
in the global train+dev wav.scp, measure a noise-floor estimate and either denoise (DeepFilterNet,
noise >= -40dBFS) or pass through untouched (noise < -40dBFS, already clean) -- resampling to
24kHz in both cases -- and write the result to a mirrored tree under speech/denoised/, replacing
"speech_out" with "denoised" in the path so per-corpus structure is preserved.

Noise-floor estimate: 20ms-frame RMS in dBFS, 10th percentile across frames -- a cheap, standard
proxy for the noise floor that doesn't need VAD. Runs entirely on CPU (DeepFilterNet is real-time
capable even without a GPU), so this competes with nothing GPU-bound.

Sharded the same way as 02_extract_features_shard.py (--shard-idx/--num-shards over the same
deterministic sorted utterance list) for consistency with the rest of the pipeline and so this
job class can run alongside it without fighting over node types.
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", message=".*has been moved to.*")

GLOBAL_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data/global")
MANIFEST_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/denoise_manifests")
SPEECH_OUT_ROOT = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out"
DENOISED_ROOT = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/denoised"

NOISE_THRESHOLD_DB = -40.0
TARGET_SR = 24000


def load_combined_wav_scp():
    utt2wav = {}
    for split in ("train", "dev"):
        with open(GLOBAL_DIR / split / "wav.scp") as f:
            for line in f:
                utt, wav = line.rstrip("\n").split(" ", 1)
                utt2wav[utt] = wav
    return utt2wav


def mirrored_path(src_path):
    if SPEECH_OUT_ROOT not in src_path:
        return None
    return src_path.replace(SPEECH_OUT_ROOT, DENOISED_ROOT, 1)


def noise_floor_db(audio, sr, frame_ms=20):
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return -100.0
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)
    return float(np.percentile(rms_db, 10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-idx", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--language", type=str, default=None,
                        help="only process utterances whose path contains /speech_out/<language>/ "
                             "(e.g. 'hausa'); manifests go to a separate dir so this doesn't collide "
                             "with the all-language sweep's shard numbering")
    args = parser.parse_args()

    manifest_dir = MANIFEST_DIR if args.language is None else MANIFEST_DIR / f"lang_{args.language}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = manifest_dir / f"shard_{args.shard_idx:05d}.jsonl"
    done_marker = manifest_dir / f"shard_{args.shard_idx:05d}.done"
    if done_marker.exists():
        print(f"shard {args.shard_idx} already done, skipping")
        return

    utt2wav = load_combined_wav_scp()
    if args.language is not None:
        needle = f"/speech_out/{args.language}/"
        utt2wav = {u: p for u, p in utt2wav.items() if needle in p}
    all_utts = sorted(utt2wav.keys())
    shard_utts = all_utts[args.shard_idx :: args.num_shards]
    print(f"shard {args.shard_idx}/{args.num_shards}: {len(shard_utts)} utterances", flush=True)

    # lazy import + model init: only pay this cost if the shard has work to do
    df_model, df_state, resampler_cache = None, None, {}

    def get_df():
        nonlocal df_model, df_state
        if df_model is None:
            import torch
            from df.enhance import init_df
            df_model, df_state, _ = init_df()
            torch.set_num_threads(os.cpu_count() or 8)
        return df_model, df_state

    def resample_np(audio, src_sr, dst_sr):
        if src_sr == dst_sr:
            return audio
        import torchaudio
        import torch
        key = (src_sr, dst_sr)
        if key not in resampler_cache:
            resampler_cache[key] = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=dst_sr)
        t = torch.from_numpy(audio).float().unsqueeze(0)
        out = resampler_cache[key](t)
        return out.squeeze(0).numpy()

    n_denoised, n_passthrough, n_failed, n_skipped_no_mirror = 0, 0, 0, 0
    t_start = time.time()

    with open(manifest_out, "w") as mf:
        for i, utt in enumerate(shard_utts):
            src_path = utt2wav[utt]
            dst_path = mirrored_path(src_path)
            if dst_path is None:
                n_skipped_no_mirror += 1
                mf.write(json.dumps({"utt": utt, "status": "skipped_no_mirror", "src": src_path}) + "\n")
                continue
            if os.path.exists(dst_path):
                continue  # idempotent: already produced by a prior (possibly interrupted) run

            try:
                audio, sr = sf.read(src_path, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                noise_db = noise_floor_db(audio, sr)
                needs_denoise = noise_db >= NOISE_THRESHOLD_DB

                if needs_denoise:
                    import torch
                    from df.enhance import enhance
                    model, state = get_df()
                    df_sr = state.sr()
                    audio_df = resample_np(audio, sr, df_sr) if sr != df_sr else audio
                    enhanced = enhance(model, state, torch.from_numpy(audio_df).float().unsqueeze(0))
                    enhanced = enhanced.squeeze(0).numpy()
                    out_audio = resample_np(enhanced, df_sr, TARGET_SR)
                    action = "denoised"
                    n_denoised += 1
                else:
                    out_audio = resample_np(audio, sr, TARGET_SR)
                    action = "passthrough"
                    n_passthrough += 1

                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                sf.write(dst_path, out_audio, TARGET_SR)
                mf.write(json.dumps({"utt": utt, "status": action, "noise_db": round(noise_db, 2), "src": src_path, "dst": dst_path}) + "\n")
            except Exception as e:
                n_failed += 1
                mf.write(json.dumps({"utt": utt, "status": "failed", "error": str(e), "src": src_path}) + "\n")

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                print(f"  [{args.shard_idx}] {i + 1}/{len(shard_utts)} "
                      f"(denoised={n_denoised} passthrough={n_passthrough} failed={n_failed}) "
                      f"{rate:.1f} utt/s", flush=True)

    print(f"shard {args.shard_idx} done: denoised={n_denoised} passthrough={n_passthrough} "
          f"failed={n_failed} no_mirror={n_skipped_no_mirror}", flush=True)
    done_marker.touch()


if __name__ == "__main__":
    main()
