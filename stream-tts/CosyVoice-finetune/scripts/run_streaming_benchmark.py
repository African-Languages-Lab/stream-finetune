"""
Demonstrates CosyVoice3's streaming inference mode on the fine-tuned Hausa/Igbo bundles.

Streaming here means stream=True on the same inference_zero_shot() call already used
elsewhere: the LLM's speech tokens are converted to audio in growing chunks (token_hop_len
starts at 2*frame_rate tokens ~2s of audio, doubles each successive chunk up to a max) as
soon as enough tokens exist, instead of waiting for the whole utterance's tokens to finish
before running the flow+vocoder once. Chunks are yielded in order and are NOT overlapping/
cumulative -- concatenating them reconstructs the full utterance.

For each language this saves:
  - nonstream_full.wav          the stream=False baseline, single call
  - stream_chunk_0.wav, _1.wav, ...   each streamed chunk as its own file
  - stream_concatenated.wav     the streamed chunks concatenated back together
  - timing.txt                  wall-clock timestamps: non-stream total time vs.
                                 streaming time-to-first-chunk and per-chunk arrival times
"""
import os
import sys
import time

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

LANGS = {
    "hausa": {
        "bundle_dir": "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_hausa_drive",
        "out_dir": "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/streaming_hausa",
        "target_text": ("Me ya sa ba za mu iya karɓar abinci da katinmu na intane ba? "
                         "Kwararrun marubutan ba su da kuɗin shiga na yau da kullum.<|endofprompt|>"),
        "prompt_wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/hausa/audio/TWB-Voice-1.0 Hausa - hau/30333.wav",
        "prompt_text": "Binta ta shawo kan Kabiru ta canza shawara.<|endofprompt|>",
    },
    "igbo": {
        "bundle_dir": "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_igbo_drive",
        "out_dir": "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/streaming_igbo",
        "target_text": ("Casablanca bụ otu n'ime opekata mpe ebe na-adọrọ mmasị ịzụ ahịa na Morocco niile. "
                         "Eid al-Fitr bụ oge oriri, dịka ndị Alakụba na-agbaji ọnụ ọnwa ha na-ezukọ n'ekpere na mkpakọrịta.<|endofprompt|>"),
        "prompt_wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/igbo/audio/Waxal_NLP_Igbo/ibo_1248.wav",
        "prompt_text": "Iri na otu agbakọnyere atọ pụtara iri na anọ.<|endofprompt|>",
    },
}


def main():
    for lang, cfg in LANGS.items():
        print(f"\n=== {lang} ===", flush=True)
        os.makedirs(cfg["out_dir"], exist_ok=True)
        model = CosyVoice3(cfg["bundle_dir"], fp16=False)
        sr = model.sample_rate
        log_lines = []

        # --- non-streaming baseline (retry on degenerate/collapsed generation) ---
        for attempt in range(10):
            t0 = time.time()
            results = list(model.inference_zero_shot(
                cfg["target_text"], cfg["prompt_text"], cfg["prompt_wav"], stream=False))
            t_total_nonstream = time.time() - t0
            audio = results[0]["tts_speech"]
            dur = audio.shape[1] / sr
            if dur >= 3.0:
                break
            print(f"  non-stream attempt {attempt + 1} collapsed ({dur:.2f}s), retrying", flush=True)
        rtf_nonstream = t_total_nonstream / dur
        torchaudio.save(f"{cfg['out_dir']}/nonstream_full.wav", audio, sr)
        line = f"non-stream: total wall-clock {t_total_nonstream:.3f}s for {dur:.2f}s of audio -> RTF={rtf_nonstream:.3f} (compute_time / audio_duration; <1.0 = faster than real time)"
        print(line, flush=True)
        log_lines.append(line)

        # --- streaming (retry on degenerate/collapsed generation) ---
        for attempt in range(10):
            t0 = time.time()
            t_prev = 0.0
            chunk_times = []
            chunks = []
            for i, out in enumerate(model.inference_zero_shot(
                    cfg["target_text"], cfg["prompt_text"], cfg["prompt_wav"], stream=True)):
                t_arrival = time.time() - t0
                chunk_audio = out["tts_speech"]
                chunk_dur = chunk_audio.shape[1] / sr
                # incremental RTF: compute time spent producing THIS chunk / this chunk's audio length
                # (matches how the library's own internal logging defines rtf per yield)
                chunk_rtf = (t_arrival - t_prev) / chunk_dur if chunk_dur > 0 else float("nan")
                t_prev = t_arrival
                chunk_times.append((i, t_arrival, chunk_dur, chunk_rtf))
                chunks.append(chunk_audio)
            t_total_stream = time.time() - t0
            cat_dur = sum(c.shape[1] for c in chunks) / sr
            if cat_dur >= 3.0:
                break
            print(f"  stream attempt {attempt + 1} collapsed ({cat_dur:.2f}s total), retrying", flush=True)

        for i, chunk_audio in enumerate(chunks):
            torchaudio.save(f"{cfg['out_dir']}/stream_chunk_{i}.wav", chunk_audio, sr)
            _, t_arrival, chunk_dur, chunk_rtf = chunk_times[i]
            line = f"  chunk {i}: arrived at t={t_arrival:.3f}s, contains {chunk_dur:.2f}s of audio, incremental RTF={chunk_rtf:.3f}"
            print(line, flush=True)
            log_lines.append(line)

        concatenated = torch.cat(chunks, dim=1)
        torchaudio.save(f"{cfg['out_dir']}/stream_concatenated.wav", concatenated, sr)
        cat_dur = concatenated.shape[1] / sr
        rtf_stream_overall = t_total_stream / cat_dur

        line = f"stream: {len(chunks)} chunks, time-to-first-chunk={chunk_times[0][1]:.3f}s, total wall-clock {t_total_stream:.3f}s for {cat_dur:.2f}s of audio -> overall RTF={rtf_stream_overall:.3f}"
        print(line, flush=True)
        log_lines.append(line)
        speedup = t_total_nonstream - chunk_times[0][1]
        line = f"=> first audio available {speedup:.3f}s sooner with streaming than waiting for the full non-stream call"
        print(line, flush=True)
        log_lines.append(line)
        line = f"=> time-to-first-chunk RTF (first chunk only) = {chunk_times[0][3]:.3f}, vs non-stream whole-utterance RTF = {rtf_nonstream:.3f}"
        print(line, flush=True)
        log_lines.append(line)

        with open(f"{cfg['out_dir']}/timing.txt", "w") as f:
            f.write("\n".join(log_lines) + "\n")

        del model
        torch.cuda.empty_cache()

    print("\nall done", flush=True)


if __name__ == "__main__":
    main()
