"""Find a prompt form that satisfies CosyVoice3 without the reference being read aloud.

inference_cross_lingual is not an option: CosyVoice3 asserts on a missing <|endofprompt|>
and cross-lingual passes no prompt_text at all. So the question is what to put BEFORE the
marker. Compares:

  full     the whole reference transcript + marker      (current behaviour, reads the reference)
  marker   the marker alone                             (nothing to read)
  short    first three words + marker                   (middle ground)

A leak shows up as output far longer than the requested sentence warrants, so duration
against a fixed target text is the measure.
"""
import glob, json, os, sys
REPO = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))
import numpy as np, torch, soundfile as sf

PRE="/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
CK="/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual"
EOP="<|endofprompt|>"

def build(code, wd):
    import shutil
    d=os.path.join(wd,code); os.makedirs(d,exist_ok=True)
    def newest(p):
        fs=[x for x in glob.glob(f"{p}/*.pt") if os.path.basename(x)!="init.pt"]
        return max(fs,key=os.path.getmtime) if fs else None
    for stage,name in (("llm","llm.pt"),("flow","flow.pt")):
        src=newest(f"{CK}/individual_{code}/{stage}")
        sd=torch.load(src,map_location="cpu",weights_only=True)
        torch.save({k:v for k,v in sd.items() if torch.is_tensor(v)}, f"{d}/{name}")
        del sd
    for a in ("cosyvoice3.yaml","campplus.onnx","speech_tokenizer_v3.onnx","hift.pt"):
        shutil.copy2(f"{PRE}/{a}", f"{d}/{a}")
    if not os.path.exists(f"{d}/CosyVoice-BlankEN"):
        shutil.copytree(f"{PRE}/CosyVoice-BlankEN", f"{d}/CosyVoice-BlankEN")
    return d

if __name__=="__main__":
    lang, code, ref, transcript, target, out = sys.argv[1:7]
    from cosyvoice.cli.cosyvoice import CosyVoice3
    wd="/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/pvar"
    os.makedirs(wd,exist_ok=True); os.makedirs(out,exist_ok=True)
    m=CosyVoice3(build(code,wd), fp16=False); sr=int(m.sample_rate)
    words=transcript.split()
    variants={"full":   transcript.strip()+EOP,
              "marker": EOP,
              "short":  " ".join(words[:3])+EOP}
    print(f"target text ({len(target.split())} words): {target}")
    print(f"reference transcript ({len(words)} words): {transcript[:70]}\n")
    res={}
    for name,ptext in variants.items():
        durs=[]
        for _ in range(3):
            try:
                r=list(m.inference_zero_shot(target, ptext, ref, stream=False))
                w=r[0]["tts_speech"].squeeze(0).cpu().numpy().astype(np.float32)
                durs.append(round(w.shape[-1]/sr,2))
                sf.write(os.path.join(out,f"{lang}_{name}.wav"), w, sr)
            except Exception as e:
                durs.append(f"ERR {type(e).__name__}: {str(e)[:70]}")
        res[name]=durs
        print(f"  {name:7} durations={durs}")
    json.dump(res, open(os.path.join(out,f"{lang}_variants.json"),"w"), indent=2)
