"""
Extracts speaker embedding (CPU, campplus ONNX) and discrete speech token (GPU, whisper
mel + ONNX) for one shard of the global deduplicated utterance set. Runs both extractors
inline against the combined train+dev wav.scp rather than materializing per-shard Kaldi
directories, and writes shard-local utt2embedding.pt / utt2speech_token.pt. A separate
merge step (03_merge_features.py) unions all shards and computes the true global
spk2embedding average -- per-shard speaker averages would be wrong wherever a pseudo-speaker's
utterances land in more than one shard, which is common (226 speakers across ~12.9M
utterances split into many shards).

Logic mirrors CosyVoice's own tools/extract_embedding.py and tools/extract_speech_token.py
directly (same ONNX sessions, same fbank/whisper-mel feature extraction) so the outputs are
byte-for-byte what the reference pipeline would produce; only the sharding and merge-safe
output format are custom.
"""
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper

GLOBAL_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data/global")
FEATURES_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/features")
CHECKPOINT_DIR = Path("/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B")


def load_combined_wav_scp():
    utt2wav = {}
    for split in ("train", "dev"):
        with open(GLOBAL_DIR / split / "wav.scp") as f:
            for line in f:
                utt, wav = line.rstrip("\n").split(" ", 1)
                utt2wav[utt] = wav
    return utt2wav


def embedding_job(utt, wav_path, ort_session):
    # A single corrupt/empty/near-zero-length audio file must not take down the whole
    # shard: confirmed in production that this happens often enough across ~12.9M files to
    # fail essentially every shard (kaldi.fbank asserts on a zero-length waveform) -- catch
    # and skip per utterance instead, matching how speech_token_job already skips >30s audio.
    try:
        audio, sr = torchaudio.load(wav_path)
        if sr != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = ort_session.run(
            None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()}
        )[0].flatten().tolist()
        return utt, embedding
    except Exception as e:
        logging.warning("embedding_job failed for %s (%s): %s", utt, wav_path, e)
        return utt, None


def speech_token_job(utt, wav_path, ort_session):
    try:
        audio, sr = torchaudio.load(wav_path, backend="soundfile")
        if sr != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.shape[1] == 0:
            logging.warning("skipping zero-length audio: %s", wav_path)
            return utt, None
        if audio.shape[1] / 16000 > 30:
            logging.warning("skipping >30s audio that should have been pre-filtered: %s", wav_path)
            return utt, []
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        speech_token = ort_session.run(
            None,
            {
                ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
            },
        )[0].flatten().tolist()
        return utt, speech_token
    except Exception as e:
        logging.warning("speech_token_job failed for %s (%s): %s", utt, wav_path, e)
        return utt, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-idx", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--num-thread", type=int, default=16)
    args = parser.parse_args()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    emb_out = FEATURES_DIR / f"shard_{args.shard_idx:05d}.utt2embedding.pt"
    tok_out = FEATURES_DIR / f"shard_{args.shard_idx:05d}.utt2speech_token.pt"
    done_marker = FEATURES_DIR / f"shard_{args.shard_idx:05d}.done"
    if done_marker.exists():
        print(f"shard {args.shard_idx} already done, skipping")
        return

    utt2wav = load_combined_wav_scp()
    all_utts = sorted(utt2wav.keys())  # stable order so shard membership is deterministic
    shard_utts = all_utts[args.shard_idx::args.num_shards]
    print(f"shard {args.shard_idx}/{args.num_shards}: {len(shard_utts)} utterances")

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1

    emb_session = onnxruntime.InferenceSession(
        str(CHECKPOINT_DIR / "campplus.onnx"), sess_options=option, providers=["CPUExecutionProvider"]
    )
    utt2embedding = {}
    n_emb_failed = 0
    with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
        futures = [executor.submit(embedding_job, utt, utt2wav[utt], emb_session) for utt in shard_utts]
        for fut in as_completed(futures):
            utt, emb = fut.result()
            if emb is None:
                n_emb_failed += 1
                continue
            utt2embedding[utt] = emb
    torch.save(utt2embedding, emb_out)
    print(f"shard {args.shard_idx}: embeddings done ({len(utt2embedding)}, {n_emb_failed} failed)")

    tok_session = onnxruntime.InferenceSession(
        str(CHECKPOINT_DIR / "speech_tokenizer_v3.onnx"), sess_options=option, providers=["CUDAExecutionProvider"]
    )
    utt2speech_token = {}
    n_tok_failed = 0
    with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
        futures = [executor.submit(speech_token_job, utt, utt2wav[utt], tok_session) for utt in shard_utts]
        for fut in as_completed(futures):
            utt, tok = fut.result()
            if tok is None:
                n_tok_failed += 1
                continue
            utt2speech_token[utt] = tok
    torch.save(utt2speech_token, tok_out)
    print(f"shard {args.shard_idx}: speech tokens done ({len(utt2speech_token)}, {n_tok_failed} failed)")

    done_marker.touch()


if __name__ == "__main__":
    main()
