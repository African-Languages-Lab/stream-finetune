"""Listen set for the LATEST hausa, igbo and combined checkpoints.

Each model speaks English, so the three can be compared on identical text, plus its own
language where it has one. The combined model covers all three languages by definition.

Worth knowing while listening -- the CV curves say these three latest checkpoints are NOT
all equal in promise:

  hausa llm     bottomed at step 14k (cv 3.3410) and drifted up to 3.4642
  igbo  llm     bottomed at step 26k (cv 3.3892) then DIVERGED to 7.1239, 2.1x its minimum
  combined llm  flat past ~120k (3.5909 min at 206k vs 3.6022 at 123k)
  every flow    still falling at the last eval

So if the igbo output sounds worse than you remember, the loss curve predicts exactly that.
That is the point of generating from the latest rather than assuming newest means best.

Both modes are produced. Zero-shot sends the reference clip's transcript; cross-lingual sends
none. Corpus transcripts here have repeatedly failed to match their audio, which collapses
zero-shot, so cross-lingual is the fair test of the model itself.
"""
import json
import os
import shutil
import sys

REPO = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))
FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

PRE = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
CK = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints"
FINAL = os.path.join(FT, "voice_selection", "final")
OUT = "/leonardo_scratch/large/userexternal/atsado00/bestllm_latestflow"
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
EOP = "You are a helpful assistant.<|endofprompt|>"

# The three LOWEST-CV llm checkpoints for each model, each paired with the LATEST flow.
# The llms diverge -- igbo's final checkpoint sits at 2.1x its own minimum -- while every flow
# was still improving at its last eval, so the newest flow really is the best flow while the
# newest llm certainly is not. Pairing them takes the best of each rather than the newest of
# both, which is what "latest checkpoint" was silently giving us.
MODELS = {
    "hausa_llm10k":  (f"{CK}/individual/individual_ha-NG", "epoch_0_step_10000.pt", "epoch_7_whole.pt"),
    "hausa_llm50k":  (f"{CK}/individual/individual_ha-NG", "epoch_1_step_50000.pt", "epoch_7_whole.pt"),
    "hausa_llm60k":  (f"{CK}/individual/individual_ha-NG", "epoch_2_step_60000.pt", "epoch_7_whole.pt"),
    "igbo_llm30k":   (f"{CK}/individual/individual_ig-NG", "epoch_3_step_30000.pt", "epoch_23_whole.pt"),
    "igbo_llm10k":   (f"{CK}/individual/individual_ig-NG", "epoch_1_step_10000.pt", "epoch_23_whole.pt"),
    "igbo_llm20k":   (f"{CK}/individual/individual_ig-NG", "epoch_2_step_20000.pt", "epoch_23_whole.pt"),
    "combined_llm210k": (f"{CK}/combined/combined", "epoch_10_step_210000.pt", "epoch_12_step_250000.pt"),
    "combined_llm230k": (f"{CK}/combined/combined", "epoch_11_step_230000.pt", "epoch_12_step_250000.pt"),
    "combined_llm200k": (f"{CK}/combined/combined", "epoch_9_step_200000.pt", "epoch_12_step_250000.pt"),
}
# which languages each model is asked to speak
def speaks(name):
    if name.startswith("hausa"): return ["english", "hausa"]
    if name.startswith("igbo"): return ["english", "igbo"]
    return ["english", "hausa", "igbo"]

TEXT = {
 "english": ("Good morning everyone, and thank you for joining us today. Before we begin, I "
             "would like to remind you that the meeting is being recorded, and the slides "
             "will be shared with everyone afterwards. We have three items on the agenda."),
 "hausa":   ("Sannu da safe, yaya kake? Ina fatan kana lafiya a yau. Ilimi shine mabudin ci "
             "gaba a kowace kasa ta duniya. Yara maza da mata su tafi makaranta domin su koyi "
             "karatu da rubutu."),
 "igbo":    ("Ndewo, kedu ka i mere? Enwere m olileanya na i di mma taa. Agumakwukwo bu isi "
             "ihe na-eweta oganihu na obodo obula. Umuaka nwoke na nwanyi kwesiri iga akwukwo."),
}
# the reference clip lends its voice; its language need not match what is spoken
REF = {"english": "hausa", "hausa": "hausa", "igbo": "igbo"}


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def build(base, llm, flow, d):
    os.makedirs(d, exist_ok=True)
    clean(f"{base}/llm/{llm}", f"{d}/llm.pt")
    clean(f"{base}/flow/{flow}", f"{d}/flow.pt")
    for a in SHARED:
        shutil.copy2(f"{PRE}/{a}", f"{d}/{a}")
    if not os.path.exists(f"{d}/CosyVoice-BlankEN"):
        shutil.copytree(f"{PRE}/CosyVoice-BlankEN", f"{d}/CosyVoice-BlankEN")
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    from cosyvoice.cli.cosyvoice import CosyVoice3
    prompts = json.load(open(os.path.join(FINAL, "prompts.json")))

    print(f"{'model':11}{'ckpt':44}{'speaks':9}{'gender':8}{'mode':6}{'dur':>8}", flush=True)
    for name, (base, llm, flow) in MODELS.items():
        d = os.path.join(OUT, "work", name)
        try:
            model = CosyVoice3(build(base, llm, flow, d), fp16=False)
        except Exception as e:
            print(f"{name}: BUILD FAILED {str(e)[:70]}", flush=True); continue
        tag = f"{llm.replace('.pt','')}+{flow.replace('.pt','')}"
        for lang in speaks(name):
            text = TEXT[lang]
            refl = REF[lang]
            for g in ("male", "female"):
                e = prompts[refl][g]
                wav = os.path.join(FINAL, e["prompt_wav"])
                for mode in ("xl",):
                    try:
                        r = (model.inference_zero_shot(text, EOP + e["prompt_text"], wav, stream=False)
                             if mode == "zs" else
                             model.inference_cross_lingual(EOP + text, wav, stream=False))
                        au = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy() for x in list(r)])
                        p = f"{OUT}/{name}__{lang}__{g}__{mode}.wav"
                        sf.write(p, au, 24000)
                        print(f"{name:11}{tag[:43]:44}{lang:9}{g:8}{mode:6}{len(au)/24000:7.2f}s", flush=True)
                    except Exception as ex:
                        print(f"{name:11}{tag[:43]:44}{lang:9}{g:8}{mode:6}  ERROR {str(ex)[:40]}", flush=True)
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(d, ignore_errors=True)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
