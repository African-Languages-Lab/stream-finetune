"""Find a Yoruba checkpoint pair that actually speaks.

Yoruba was written off earlier as "output collapses to under a second", but that verdict came
from a single checkpoint choice. There is no tensorboard history for yo-NG, so the usual
best() helper falls back to "newest", which is a guess rather than a selection -- the last
epoch of a run that may have diverged. So sweep a grid instead: several LLM epochs against
several flow epochs, all with the ORIGINAL pretrained vocoder.

Also checks the tokenizer first. The 277-token vocab expansion missed Yoruba, so its text may
be falling back to byte-level pieces. If it does, every checkpoint is handicapped by the input
representation and no amount of checkpoint-picking will fix it -- worth knowing before
reading anything into the audio.
"""
import json
import os
import shutil
import sys
import traceback

REPO = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

PRETRAINED = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
CKPT = os.environ.get("SWEEP_CKPT",
        "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_yo-NG")
FT = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune"
WAVDIR = os.path.join(FT, "voice_selection", "eval14_wavs")
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
OUT = os.environ.get("SWEEP_OUT", "/leonardo_scratch/large/userexternal/atsado00/yoruba_sweep")
WORK = os.path.join(OUT, "work")

EOP = "You are a helpful assistant.<|endofprompt|>"

REFS = {
    "male": ("yoruba__male_1.wav",
             "ọrọ̀ yìí “bug” ní àwọn onimọ kòkòrò máa ń lò, láti pe èyíkéyìí àkójọpọ̀ àwọn kòkòrò."),
    "female": ("yoruba__female_2.wav",
               "Awọn obinrin mu awọn ipa oriṣiriṣi mu ni awujọ, gẹgẹbi awọn iya, awọn oludari, "
               "awọn olukọni, ati awọn alamọdaju."),
}
TARGET = "Ẹ káàárọ̀, ṣé àlàáfíà ni? Mo ń retí pé ara yín le dáadáa lónìí."

LLMS = os.environ.get("SWEEP_LLMS", "epoch_1_whole.pt,epoch_3_whole.pt,epoch_5_whole.pt,epoch_7_whole.pt").split(",")
FLOWS = os.environ.get("SWEEP_FLOWS", "epoch_10_whole.pt,epoch_20_whole.pt,epoch_30_whole.pt").split(",")


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def build(llm, flow, d):
    os.makedirs(d, exist_ok=True)
    clean(f"{CKPT}/llm/{llm}", f"{d}/llm.pt")
    clean(f"{CKPT}/flow/{flow}", f"{d}/flow.pt")
    for a in SHARED:                       # hift.pt here is the ORIGINAL pretrained vocoder
        shutil.copy2(f"{PRETRAINED}/{a}", f"{d}/{a}")
    dst = f"{d}/CosyVoice-BlankEN"
    if not os.path.exists(dst):
        shutil.copytree(f"{PRETRAINED}/CosyVoice-BlankEN", dst)
    return d


def tokenizer_report():
    """Does Yoruba survive the tokenizer, or is it shredded into bytes?"""
    from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
    tk = get_qwen_tokenizer(token_path=f"{PRETRAINED}/CosyVoice-BlankEN", skip_special_tokens=True)
    print("\n=== tokenizer coverage ===", flush=True)
    samples = {
        "yoruba": TARGET,
        "yoruba_ref": REFS["male"][1],
        "hausa (works)": "Sannu da safe, yaya kake? Ina fatan kana lafiya a yau.",
        "swahili (works)": "Habari ya asubuhi, hujambo? Natumaini u mzima leo.",
    }
    for name, txt in samples.items():
        ids = tk.encode(txt, allowed_special="all")
        chars = max(len(txt), 1)
        print(f"  {name:16} {chars:4d} chars -> {len(ids):4d} tokens  "
              f"({len(ids) / chars:.2f} tokens/char)", flush=True)
    print("  (a much higher tokens/char than the working languages means Yoruba text is\n"
          "   being split into byte-level pieces the model never learned as words)", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    from cosyvoice.cli.cosyvoice import CosyVoice3

    try:
        tokenizer_report()
    except Exception as e:
        print("tokenizer check failed:", str(e)[:200], flush=True)

    results = []
    print(f"\n=== sweep: {len(LLMS)} llm x {len(FLOWS)} flow, original vocoder ===", flush=True)
    print(f"{'llm':10}{'flow':12}{'ref':8}{'dur':>7}{'exp':>7}{'peak':>7}  verdict", flush=True)
    for llm in LLMS:
        for flow in FLOWS:
            tag = f"{llm.replace('.pt','')}__{flow.replace('.pt','')}"
            d = os.path.join(WORK, tag)
            try:
                build(llm, flow, d)
                model = CosyVoice3(d, fp16=False)
            except Exception as e:
                print(f"{llm[:9]:10}{flow[:11]:12}{'-':8}  BUILD FAILED {str(e)[:60]}", flush=True)
                continue
            for g, (wav, ptext) in REFS.items():
                wp = os.path.join(WAVDIR, wav)
                ref_dur = sf.info(wp).duration
                expected = len(TARGET) / (len(ptext) / ref_dur)
                try:
                    res = list(model.inference_zero_shot(
                        TARGET, EOP + ptext, wp, stream=False))
                    au = np.concatenate([r["tts_speech"].squeeze(0).cpu().numpy() for r in res])
                except Exception as e:
                    print(f"{llm[:9]:10}{flow[:11]:12}{g:8}  ERROR {str(e)[:60]}", flush=True)
                    continue
                dur = len(au) / 24000
                pk = float(np.abs(au).max()) if au.size else 0.0
                verdict = ("COLLAPSED" if dur < expected * 0.4 else
                           "SILENT" if pk < 0.02 else
                           "LONG" if dur > expected * 2.2 else "OK")
                p = os.path.join(OUT, f"{tag}__{g}.wav")
                sf.write(p, au, 24000)
                results.append(dict(llm=llm, flow=flow, ref=g, dur=round(dur, 2),
                                    expected=round(expected, 2), peak=round(pk, 3),
                                    verdict=verdict, wav=os.path.basename(p)))
                print(f"{llm[:9]:10}{flow[:11]:12}{g:8}{dur:7.2f}{expected:7.1f}{pk:7.3f}  {verdict}",
                      flush=True)
            del model
            torch.cuda.empty_cache()
            shutil.rmtree(d, ignore_errors=True)
            json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)

    ok = [r for r in results if r["verdict"] == "OK"]
    print(f"\n=== {len(ok)} of {len(results)} generations usable ===", flush=True)
    for r in sorted(ok, key=lambda x: abs(x["dur"] - x["expected"]))[:10]:
        print(f"  {r['llm']} + {r['flow']} [{r['ref']}] {r['dur']}s (exp {r['expected']}s)", flush=True)


if __name__ == "__main__":
    main()
