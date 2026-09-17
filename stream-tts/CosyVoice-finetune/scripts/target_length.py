"""Is the collapse caused by short input text rather than by the prompt form?

The prompt-variant run collapsed on all three forms with a 5-word target, while the same
voice produced 4-5s of clean speech on an 11-word target. CosyVoice warns when the target
is under half the length of the reference transcript, so this crosses target length against
prompt form to see which actually drives it.
"""
import glob, json, os, sys
REPO="/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice"
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"third_party","Matcha-TTS"))
import numpy as np, torch, soundfile as sf
PRE="/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
CK="/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual"
EOP="<|endofprompt|>"

TARGETS=[("very short","Sannu."),
         ("short","Sannu da safe."),
         ("medium","Sannu da safe, yaya kake?"),
         ("normal","Sannu da safe, yaya kake? Ina fatan kana lafiya a yau."),
         ("long","Sannu da safe, yaya kake? Ina fatan kana lafiya a yau. Muna farin cikin ganin ku a nan.")]

def build(code,wd):
    import shutil
    d=os.path.join(wd,code); os.makedirs(d,exist_ok=True)
    def newest(p):
        fs=[x for x in glob.glob(f"{p}/*.pt") if os.path.basename(x)!="init.pt"]
        return max(fs,key=os.path.getmtime) if fs else None
    for stage,name in (("llm","llm.pt"),("flow","flow.pt")):
        sd=torch.load(newest(f"{CK}/individual_{code}/{stage}"),map_location="cpu",weights_only=True)
        torch.save({k:v for k,v in sd.items() if torch.is_tensor(v)}, f"{d}/{name}"); del sd
    for a in ("cosyvoice3.yaml","campplus.onnx","speech_tokenizer_v3.onnx","hift.pt"):
        shutil.copy2(f"{PRE}/{a}", f"{d}/{a}")
    if not os.path.exists(f"{d}/CosyVoice-BlankEN"):
        shutil.copytree(f"{PRE}/CosyVoice-BlankEN", f"{d}/CosyVoice-BlankEN")
    return d

if __name__=="__main__":
    ref, transcript, out = sys.argv[1:4]
    from cosyvoice.cli.cosyvoice import CosyVoice3
    wd="/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/tlen"; os.makedirs(wd,exist_ok=True)
    os.makedirs(out,exist_ok=True)
    m=CosyVoice3(build("ha-NG",wd),fp16=False); sr=int(m.sample_rate)
    forms={"full":transcript.strip()+EOP, "marker":EOP,
           "short_prompt":" ".join(transcript.split()[:4])+EOP}
    print(f"reference transcript: {len(transcript.split())} words / {len(transcript)} chars\n")
    res={}
    print(f"{'target':11} {'chars':>6}  " + "  ".join(f"{k:22}" for k in forms))
    print("-"*80)
    for label,tt in TARGETS:
        row=[]
        for fname,ptext in forms.items():
            durs=[]
            for _ in range(2):
                try:
                    r=list(m.inference_zero_shot(tt,ptext,ref,stream=False))
                    w=r[0]["tts_speech"].squeeze(0).cpu().numpy().astype(np.float32)
                    durs.append(round(w.shape[-1]/sr,2))
                    sf.write(os.path.join(out,f"{label.replace(' ','_')}_{fname}.wav"),w,sr)
                except Exception as e:
                    durs.append(f"ERR:{type(e).__name__}")
            row.append(str(durs)); res[f"{label}|{fname}"]=durs
        print(f"{label:11} {len(tt):>6}  " + "  ".join(f"{c:22}" for c in row))
    json.dump(res,open(os.path.join(out,"target_length.json"),"w"),indent=2)
