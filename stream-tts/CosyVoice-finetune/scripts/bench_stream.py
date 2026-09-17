"""Time the SAME streaming request on Leonardo's A100 that the endpoint serves on an A10G.

The endpoint measures RTF 0.94; 424 logged runs on Leonardo averaged 0.47. Both use identical
flags -- CosyVoice3(dir, fp16=False), no TensorRT, no vLLM -- so the gap should be hardware
alone. This runs the identical text and reference through the identical code path to confirm
that, rather than comparing two measurements that were never meant to be compared.

Reports the same numbers measure_stream.py reports against the live endpoint: time to first
audio, real-time factor, and total wall clock.
"""
import os
import sys
import time

REPO = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))
FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(FT, "scripts"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from try_new_models import build  # noqa: E402

EOP = "You are a helpful assistant.<|endofprompt|>"
FINAL = os.path.join(FT, "voice_selection", "final")
OUT = "/leonardo_scratch/large/userexternal/atsado00/benchstream"

S = ("Sannu da safe, yaya kake? Ina fatan kana lafiya a yau. "
     "Ilimi shine mabudin ci gaba a kowace kasa ta duniya. "
     "Yara maza da mata su tafi makaranta domin su koyi karatu da rubutu. "
     "Malamai suna aiki tukuru domin su ilimantar da al'umma. ")
TEXT = (S * 12).strip()          # the same 2,783-character passage used on the endpoint


def main():
    os.makedirs(OUT, exist_ok=True)
    import json
    from cosyvoice.cli.cosyvoice import CosyVoice3

    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)
    prompts = json.load(open(os.path.join(FINAL, "prompts.json")))["hausa"]["male"]
    wav = os.path.join(FINAL, prompts["prompt_wav"])
    ptext = EOP + prompts["prompt_text"]

    d = build("ha-NG", os.path.join(OUT, "work"))
    model = CosyVoice3(d, fp16=False)

    # one short warm-up so the timed run measures steady state, as the endpoint's does
    list(model.inference_zero_shot("Sannu.", ptext, wav, stream=False))

    print(f"\ntext: {len(TEXT)} chars", flush=True)
    t0 = time.time(); ttfa = None; chunks = []
    for out in model.inference_zero_shot(TEXT, ptext, wav, stream=True):
        a = out["tts_speech"].squeeze(0).cpu().numpy()
        if a.size == 0:
            continue
        if ttfa is None:
            ttfa = time.time() - t0
        chunks.append(a)
    total = time.time() - t0
    au = np.concatenate(chunks)
    audio = len(au) / 24000
    sf.write(os.path.join(OUT, "leonardo_a100_4min.wav"), au, 24000)

    print(f"\n  TTFA   {ttfa:7.2f}s")
    print(f"  TOTAL  {total:7.2f}s   for {audio:.1f}s of audio")
    print(f"  RTF    {total / audio:7.2f}", flush=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "chars": len(TEXT),
               "ttfa": round(ttfa, 2), "total": round(total, 2),
               "audio_sec": round(audio, 1), "rtf": round(total / audio, 3)},
              open(os.path.join(OUT, "result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
