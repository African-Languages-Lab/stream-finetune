"""Synthesize with newly-finished individual models to see whether they are worth serving.

Builds each model directory locally from the trained checkpoints plus the pretrained shared
assets, so this runs on a compute node with no outbound network (Leonardo's GPU nodes cannot
reach the Hub).

Judges the output the way the earlier audit did -- a model is only worth serving if it
produces full-length speech at a healthy level, rather than the fraction-of-a-second collapse
that disqualified af-ZA, en-UG, ki-KE, nd-ZW, rw-RW and yo-NG.

Uses zero-shot inference with the reference transcript. CosyVoice3 REQUIRES the
<|endofprompt|> marker -- inference_cross_lingual omits prompt_text entirely and therefore
asserts out ("<|endofprompt|> not detected"), so cross-lingual is simply not available for
this model. The marker is what tells the LLM where the reference ends, so a correct one is
also what stops the reference being read aloud.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import traceback

REPO = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

PRETRAINED = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
CKPT_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual"
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]

# a plain sentence per language, long enough that a collapse is unmistakable
PROBES = {
    "swati": "Sawubona, unjani namuhla? Ngiyetsemba kutsi uphila kahle.",
    "bambara": "I ni sɔgɔma, i ka kɛnɛ wa? N b'a fɛ ka a fɔ ko i ka kɛnɛ.",
    "kanuri": "Ndaraa, awo nyi ci? Nyi kalak ba wa.",
    "fon": "A fɔn ganji a? Un ɖo nukún ɖɔ a ɖo ganji egbé.",
    "krio": "Kushɛ, aw yu du? A op se yu de fayn tide.",
    "afrikaans": "Goeie môre, hoe gaan dit met jou? Ek hoop jy is vandag gesond.",
    "bemba": "Mwashibukeni, muli shani lelo? Ndesubila ukuti muli bwino.",
    "english": "Good morning, how are you today? I hope you are keeping well.",
    "kikuyu": "Wĩ mwega? Nĩ ndĩrehoka atĩ ũrĩ mwega ũmũthĩ.",
    "ndebele": "Sawubona, unjani lamuhla? Ngithemba ukuthi uphilile.",
    "kinyarwanda": "Mwaramutse, amakuru yawe? Nizeye ko umeze neza uyu munsi.",
    "shona": "Mangwanani, wakadii nhasi? Ndinovimba uri kunzwa zvakanaka.",
    "wolof": "Jaam nga fanaan, naka nga def? Dama yaakaar ne jamm rekk nga am.",
    "yoruba": "Ẹ káàárọ̀, ṣé àlàáfíà ni? Mo ń retí pé ara yín le dáadáa lónìí.",
    # the 24 already served -- re-verified against their deployed reference clips
    "hausa": "Sannu da safe, yaya kake? Ina fatan kana lafiya a yau.",
    "twi": "Maakye, wo ho te sɛn nnɛ? Mewɔ anidasoɔ sɛ wo ho yɛ.",
    "igbo": "Ndewo, kedu ka i mere? Enwere m olileanya na i di mma taa.",
    "ewe": "Ŋdi na wò, aleke nèfɔ egbea? Mele mɔ kpɔm be èle nyuie.",
    "berber": "Azul, amek tettiliḍ ass-a? Ssarameɣ tellid di lehna.",
    "umbundu": "Wakolapo, ove wa kala ndati etaili? Ndi kolela okuti wa kola.",
    "amharic": "እንደምን አደርክ? ዛሬ ደህና እንደሆንክ ተስፋ አደርጋለሁ።",
    "arabic": "صباح الخير، كيف حالك اليوم؟ أتمنى أن تكون بخير.",
    "fula": "Jam waali, no mbaɗ-ɗaa hannde? Mi yenanaama a wonii e jam.",
    "luganda": "Wasuze otya nno? Nsuubira nti oli bulungi leero.",
    "lingala": "Mbote, ozali malamu lelo? Nazali kolikya ete ozali malamu.",
    "malagasy": "Manao ahoana ianao androany? Manantena aho fa salama tsara ianao.",
    "sepedi": "Thobela, o kae lehono? Ke holofela o phela gabotse.",
    "chichewa": "Mwadzuka bwanji? Ndikukhulupirira kuti muli bwino lero.",
    "oromo": "Akkam jirta har'a? Nagaa akka jirtu abdadha.",
    "somali": "Subax wanaagsan, sidee tahay maanta? Waxaan rajaynayaa inaad fiican tahay.",
    "sesotho": "Dumela, o phela joang kajeno? Ke tshepa o phetse hantle.",
    "swahili": "Habari ya asubuhi, hujambo? Natumaini u mzima leo.",
    "tigrinya": "ከመይ ሓዲርካ? ሎሚ ጽቡቕ ከም ዘለኻ ተስፋ እገብር።",
    "tswana": "Dumela, o tsogile jang gompieno? Ke solofela o itekanetse.",
    "tsonga": "Avuxeni, u njhani namuntlha? Ndzi tshemba u hanya kahle.",
    "venda": "Ndaa, ni hani namusi? Ndi fulufhela uri ni khou tshila zwavhudi.",
    "xhosa": "Molo, unjani namhlanje? Ndiyathemba ukuba uphilile.",
    "zulu": "Sawubona, unjani namuhla? Ngiyethemba ukuthi uphilile.",
}


def highest(stage_dir):
    fs = [p for p in glob.glob(f"{stage_dir}/*.pt") if os.path.basename(p) != "init.pt"]
    return max(fs, key=os.path.getmtime) if fs else None


def best(stage_dir):
    """Checkpoint at the lowest CV loss, matching how the earlier audit picked models.

    Falls back to the newest checkpoint when tensorboard scalars are unavailable -- the
    newest is not necessarily the best, so the fallback is reported rather than silent.
    """
    tb = stage_dir.replace("/llm", "/tensorboard/llm").replace("/flow", "/tensorboard/flow")
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(tb, size_guidance={"scalars": 0})
        ea.Reload()
        events = ea.Scalars("CV/loss")
        best_ev = min(events, key=lambda e: e.value)
        fs = [p for p in glob.glob(f"{stage_dir}/*.pt") if os.path.basename(p) != "init.pt"]
        if not fs:
            return None, None
        pick = min(fs, key=lambda p: abs(os.path.getmtime(p) - best_ev.wall_time))
        return pick, round(best_ev.value, 4)
    except Exception:
        return highest(stage_dir), None


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def build(lang_code, workdir):
    d = os.path.join(workdir, lang_code)
    os.makedirs(d, exist_ok=True)
    llm, llm_cv = best(f"{CKPT_ROOT}/individual_{lang_code}/llm")
    flow, flow_cv = best(f"{CKPT_ROOT}/individual_{lang_code}/flow")
    if not llm or not flow:
        raise FileNotFoundError(f"{lang_code}: missing llm or flow checkpoint")
    print(f"  llm  {os.path.basename(llm)}  CV/loss={llm_cv if llm_cv is not None else 'n/a (newest)'}", flush=True)
    print(f"  flow {os.path.basename(flow)}  CV/loss={flow_cv if flow_cv is not None else 'n/a (newest)'}", flush=True)
    print("  vocoder: original pretrained hift.pt (not fine-tuned)", flush=True)
    clean(llm, f"{d}/llm.pt")
    clean(flow, f"{d}/flow.pt")
    for a in SHARED:
        shutil.copy2(f"{PRETRAINED}/{a}", f"{d}/{a}")
    dst = f"{d}/CosyVoice-BlankEN"
    if not os.path.exists(dst):
        shutil.copytree(f"{PRETRAINED}/CosyVoice-BlankEN", dst)
    return d


def judge(wav, sr, text):
    dur = wav.shape[-1] / sr
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    rms = float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0
    expected = max(1.0, 0.22 * len(text.split()))
    if dur < expected * 0.35:
        return "COLLAPSED", dur, peak, rms
    if peak < 0.02:
        return "SILENT", dur, peak, rms
    return "OK", dur, peak, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="json: {lang: {male:{path,..}, female:{...}}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wavdir", required=True)
    ap.add_argument("--workdir", default="/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/try_new")
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--codes", nargs="+", required=True, help="matching checkpoint codes")
    a = ap.parse_args()
    os.makedirs(a.wavdir, exist_ok=True)
    os.makedirs(a.workdir, exist_ok=True)
    refs = json.load(open(a.refs))
    import soundfile as sf
    from cosyvoice.cli.cosyvoice import CosyVoice3

    results = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for lang, code in zip(a.langs, a.codes):
        if lang in results:
            print(f"{lang}: done already", flush=True)
            continue
        print(f"\n=== {lang}  ({code}) ===", flush=True)
        try:
            d = build(code, a.workdir)
            model = CosyVoice3(d, fp16=False)
            sr = int(model.sample_rate)
            text = PROBES[lang]
            out = {}
            for g in sorted((refs.get(lang) or {}).keys()):
                r = (refs.get(lang) or {}).get(g)
                if not r:
                    print(f"  {g}: no reference clip", flush=True)
                    continue
                wavs, verdicts = [], []
                for attempt in range(3):
                    try:
                        ptext = r.get("transcript", "").strip()
                        if not ptext.endswith("<|endofprompt|>"):
                            ptext += "<|endofprompt|>"
                        res = list(model.inference_zero_shot(text, ptext, r["path"], stream=False))
                        w = res[0]["tts_speech"].squeeze(0).cpu().numpy().astype(np.float32)
                        v, dur, peak, rms = judge(w, sr, text)
                        verdicts.append(v)
                        wavs.append((w, dur, peak))
                    except Exception as e:
                        verdicts.append(f"ERROR {type(e).__name__}: {e}")
                if wavs:
                    longest = max(wavs, key=lambda t: t[1])
                    sf.write(os.path.join(a.wavdir, f"{lang}__{g}.wav"), longest[0], sr)
                out[g] = {"verdicts": verdicts,
                          "durations": [round(d, 2) for _, d, _ in wavs],
                          "peaks": [round(p, 3) for _, _, p in wavs],
                          "usable": verdicts.count("OK") >= 2,
                          "reference": r["path"], "transcript": r.get("transcript", "")}
                print(f"  {g:7} {verdicts} durations={[round(d,2) for _,d,_ in wavs]}", flush=True)
            results[lang] = {"code": code, "voices": out,
                             "recommend": any(v.get("usable") for v in out.values())}
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            traceback.print_exc()
            results[lang] = {"code": code, "error": traceback.format_exc()[-400:], "recommend": False}
        json.dump(results, open(a.out, "w"), ensure_ascii=False, indent=2)

    print("\n============ VERDICT ============", flush=True)
    for lang, r in results.items():
        print(f"  {lang:9} {'RECOMMEND' if r.get('recommend') else 'do not serve'}", flush=True)


if __name__ == "__main__":
    main()
