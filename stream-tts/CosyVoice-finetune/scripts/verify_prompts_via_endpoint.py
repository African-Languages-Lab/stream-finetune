"""Verify candidate reference prompts by synthesizing through the live endpoint.

Leonardo's compute nodes have no outbound network, so the models cannot be pulled there;
but the endpoint already holds every model and accepts an arbitrary reference clip via
prompt_audio_base64. That makes it the natural verification harness -- and it tests the
exact model that will serve Elorm, not a local reconstruction of it.

The fault being screened for is a prompt whose transcript does not match its audio:
CosyVoice3 collapses to a fraction of a second, which no acoustic measurement detects.
Each prompt is synthesized several times because generation is stochastic; a prompt that
collapses on any run is rejected rather than averaged.

Candidates are grouped by language so each model is loaded once (a language switch costs
~2 minutes on the endpoint).
"""
import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request

URL = os.environ.get(
    "COSYVOICE_ENDPOINT",
    "https://nkqyhrnx42iu9b6d.eu-west-1.aws.endpoints.huggingface.cloud/",
)
TOKEN = os.environ.get("HF_TOKEN_ORGOWNER") or os.environ.get("HF_TOKEN")

PROBES = {
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

RUNS = 3
MIN_SEC = 1.5          # a full greeting sentence is always well above this


def call(payload, timeout=900):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode()[:200]
        except Exception:
            body = ""
        return None, f"HTTP {e.code} {body}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def verify(lang, cands):
    text = PROBES.get(lang)
    if not text:
        print(f"  {lang}: no probe sentence, skipping", flush=True)
        return []
    out = []
    for c in cands:
        wav, tag = c["prompt_wav"], c.get("tag", "")
        if not os.path.exists(wav):
            print(f"  [{lang}/{tag}] MISSING {wav}", flush=True)
            continue
        b64 = base64.b64encode(open(wav, "rb").read()).decode()
        ptext = c["prompt_text"].replace("<|endofprompt|>", "").strip()
        durs, fails, err = [], 0, None
        for _ in range(RUNS):
            d, e = call({"inputs": text, "language": lang,
                         "prompt_text": ptext, "prompt_audio_base64": b64})
            if e:
                err = e
                fails += 1
                continue
            dur = float(d.get("duration_sec", 0))
            durs.append(round(dur, 2))
            if dur < MIN_SEC or float(d.get("peak", 0)) < 0.02:
                fails += 1
        ok = bool(durs) and fails == 0
        out.append({**c, "durations": durs, "failures": fails,
                    "verified": ok, "error": err})
        print(f"  [{lang}/{tag}] durations={durs} failures={fails}"
              f"{' err=' + err if err else ''} -> {'PASS' if ok else 'REJECT'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*")
    args = ap.parse_args()

    cands = json.load(open(args.candidates))
    langs = args.langs or list(cands)

    out = {}
    if os.path.exists(args.out):
        try:
            out = json.load(open(args.out))
        except Exception:
            out = {}

    for lang in langs:
        if lang in out and out[lang]:
            print(f"{lang}: already done, skipping", flush=True)
            continue
        print(f"\n=== {lang} ===", flush=True)
        t0 = time.time()
        out[lang] = verify(lang, cands.get(lang, []))
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)

    print("\n================ SUMMARY ================", flush=True)
    for lang, rs in out.items():
        good = [r for r in rs if r.get("verified")]
        print(f"  {lang:10} {len(good)}/{len(rs)} verified", flush=True)


if __name__ == "__main__":
    main()
