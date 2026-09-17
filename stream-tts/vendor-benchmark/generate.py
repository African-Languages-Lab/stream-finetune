"""Generate the same test statements with Mansa, Pawa and Intron production TTS.

Every clip is one (vendor, language, accent, voice, variant) cell. Cells a vendor cannot
serve are written to the manifest as "unsupported" rather than silently left out, so the
comparison table shows the gap.

Keys are read from the environment or stream/.env:
    MANSA_API_KEY     (MANSA_API_BASE_URL optional, default https://api.mymansa.ai)
    PAWA_AI_API_KEY
    INTRON_API_KEY
A vendor with no key is skipped.

    python generate.py --dry-run                  # print the plan
    python generate.py                            # all vendors with keys
    python generate.py --vendor intron --lang hausa
Outputs: <out>/<vendor>/<language>/<voice>__<variant>.<ext> and <out>/manifest.jsonl.
Existing clips are skipped, so an interrupted run resumes.
"""
import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE.parents[1] / ".env"
DEFAULT_OUT = HERE.parent / "CosyVoice-finetune" / "vendor_tts_bench"
GENDERS = ("male", "female")


def load_env():
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line.strip())
            if m and m.group(1) not in os.environ:
                os.environ[m.group(1)] = m.group(2).strip().strip("'\"")


# --------------------------------------------------------------------------- plan
# English accent -> vendor voice settings. Intron has no Ghanaian English; Nigerian English
# is offered as three accents and East African as two, so each gets its own clips.
MANSA_EN = {"nigerian": ["nigerian"], "ghanaian": ["ghanaian"], "east_african": ["east_african"]}
INTRON_EN = {"nigerian": ["hausa", "igbo", "yoruba"], "ghanaian": [], "east_african": ["swahili", "luganda"]}
INTRON_LANG = {"hausa": "ha", "igbo": "ig", "yoruba": "yo", "swahili": "sw"}
PAWA_LANGS = ("english", "swahili")
# Pawa's TTS request has no language field and its TTS docs list only Swahili and English.
# Hausa, Igbo and Yoruba appear only under its speech-to-text section, Akan (Twi) only on a
# general languages page, and Ewe nowhere. They are generated anyway so listeners can judge,
# and marked "undocumented" in the manifest.
PAWA_UNDOCUMENTED = ("hausa", "igbo", "yoruba", "twi", "ewe")
PAWA_VOICES = ("ame", "liora", "ayana")


def plan(statements):
    cells = []  # dicts: vendor, language, accent, voice, variant, params | unsupported
    for lang, texts in statements.items():
        for variant, text in texts.items():
            base = {"language": lang, "variant": variant, "text": text}
            if lang == "english":
                for accent in MANSA_EN:
                    for g in GENDERS:
                        for a in MANSA_EN[accent]:
                            cells.append({**base, "vendor": "mansa", "accent": accent, "voice": f"{a}_{g}",
                                          "params": {"language": "English", "voice": f"{a}_{g}"}})
                        if INTRON_EN[accent]:
                            for a in INTRON_EN[accent]:
                                cells.append({**base, "vendor": "intron", "accent": accent, "voice": f"{a}_{g}",
                                              "params": {"voice_language": "en", "voice_accent": a, "voice_gender": g}})
                        else:
                            cells.append({**base, "vendor": "intron", "accent": accent, "voice": f"none_{g}",
                                          "unsupported": "Intron offers no Ghanaian English accent"})
                    cells.append({**base, "vendor": "pawa", "accent": accent, "voice": "-",
                                  "unsupported": "Pawa has no accent selection"})
            else:
                for g in GENDERS:
                    cells.append({**base, "vendor": "mansa", "accent": "", "voice": g,
                                  "params": {"language": lang, "voice": g}})
                    if lang in INTRON_LANG:
                        cells.append({**base, "vendor": "intron", "accent": "", "voice": g,
                                      "params": {"voice_language": INTRON_LANG[lang], "voice_accent": lang, "voice_gender": g}})
                    else:
                        cells.append({**base, "vendor": "intron", "accent": "", "voice": g,
                                      "unsupported": f"Intron does not offer {lang}"})
            if lang in PAWA_LANGS or lang in PAWA_UNDOCUMENTED:
                for v in PAWA_VOICES:
                    cells.append({**base, "vendor": "pawa", "accent": "unspecified" if lang == "english" else "",
                                  "voice": v, "params": {"voice": v},
                                  **({"support": "undocumented"} if lang in PAWA_UNDOCUMENTED else {})})
            else:
                cells.append({**base, "vendor": "pawa", "accent": "", "voice": "-",
                              "unsupported": f"Pawa supports only English and Swahili, not {lang}"})
    return cells


def out_path(root, c, ext):
    acc = f"{c['accent']}__" if c["accent"] else ""
    return root / c["vendor"] / c["language"] / f"{acc}{c['voice']}__{c['variant']}.{ext}"


# --------------------------------------------------------------------------- vendors
def _retry(fn, tries=4):
    for attempt in range(tries):
        r = fn()
        if r.status_code in (429, 502, 503, 504) and attempt < tries - 1:
            time.sleep(float(r.headers.get("Retry-After") or 30 * (attempt + 1)))
            continue
        return r
    return r


def mansa(c):
    base = os.environ.get("MANSA_API_BASE_URL", "https://api.mymansa.ai").rstrip("/")
    body = {"text": c["text"], **c["params"], "stream": False}
    t0 = time.time()
    r = _retry(lambda: requests.post(f"{base}/v1/audio/speech", json=body, timeout=900,
                                     headers={"Authorization": f"Bearer {os.environ['MANSA_API_KEY']}"}))
    if r.status_code != 200 or not r.content.startswith(b"RIFF"):
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.content, "wav", {"seconds": round(time.time() - t0, 2)}


def pawa(c):
    body = {"model": "pawa-tts-v1-20250704", "voice": c["params"]["voice"], "text": c["text"],
            "max_tokens": 65536, "temperature": 0.5, "top_p": 0.95, "repetition_penalty": 1.1}
    t0 = time.time()
    r = _retry(lambda: requests.post("https://api.pawa-ai.com/v1/voice/text-to-speech", json=body, timeout=900,
                                     headers={"Authorization": f"Bearer {os.environ['PAWA_AI_API_KEY']}"}))
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    ctype = r.headers.get("Content-Type", "")
    if not r.content:
        raise RuntimeError(f"HTTP 200 with an empty body (Content-Type {ctype!r}); nothing to save")
    if "event-stream" not in ctype:
        return r.content, "mp3" if "mpeg" in ctype or r.content[:3] in (b"ID3", b"\xff\xfb") else "bin", \
            {"seconds": round(time.time() - t0, 2), "content_type": ctype}
    # Server-Sent Events: collect audio from every data: event (base64 in JSON or bare).
    audio = bytearray()
    for line in r.text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
            data = next((obj[k] for k in ("audio", "audio_base64", "chunk", "data") if isinstance(obj, dict) and k in obj), "")
        except json.JSONDecodeError:
            pass
        if data:
            audio += base64.b64decode(data)
    if not audio:
        raise RuntimeError(f"SSE response carried no audio; first bytes: {r.text[:300]}")
    return bytes(audio), "mp3", {"seconds": round(time.time() - t0, 2), "content_type": ctype}


def _find_url(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            u = _find_url(v)
            if u:
                return u
    elif isinstance(obj, list):
        for v in obj:
            u = _find_url(v)
            if u:
                return u
    elif isinstance(obj, str) and obj.startswith("http"):
        return obj
    return None


def intron(c):
    key = os.environ["INTRON_API_KEY"]
    h = {"Authorization": f"Bearer {key}"}
    t0 = time.time()
    r = _retry(lambda: requests.post("https://infer.voice.intron.io/tts/v1/enqueue", headers=h, timeout=120,
                                     json={"text": c["text"], **c["params"], "output_audio_format": "wav"}))
    if r.status_code != 200:
        raise RuntimeError(f"enqueue HTTP {r.status_code}: {r.text[:300]}")
    text_id = r.json()["data"]["text_id"]
    while True:
        time.sleep(3)
        s = _retry(lambda: requests.get(f"https://infer.voice.intron.io/tts/v1/status/{text_id}", headers=h, timeout=60))
        j = s.json()
        status = json.dumps(j)
        if "TTS_TEXT_AUDIO_PROCESSING_FAILED" in status:
            raise RuntimeError(f"processing failed: {status[:300]}")
        if "TTS_TEXT_AUDIO_GENERATED" in status:
            break
        if time.time() - t0 > 900:
            raise RuntimeError(f"timed out waiting; last status {status[:300]}")
    url = _find_url(j.get("data", j))
    if not url:
        raise RuntimeError(f"generated but no audio URL in status response: {status[:500]}")
    a = requests.get(url, timeout=300)
    a.raise_for_status()
    return a.content, "wav" if a.content.startswith(b"RIFF") else "opus", \
        {"seconds": round(time.time() - t0, 2), "text_id": text_id}


VENDORS = {"mansa": ("MANSA_API_KEY", mansa), "pawa": ("PAWA_AI_API_KEY", pawa), "intron": ("INTRON_API_KEY", intron)}

from more_vendors import MORE_VENDORS, Unsupported, plan_more  # noqa: E402
VENDORS.update(MORE_VENDORS)


# --------------------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--vendor", action="append", choices=list(VENDORS))
    ap.add_argument("--lang", action="append")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--one-per-vendor", action="store_true", help="generate a single clip per vendor as a smoke test")
    a = ap.parse_args()
    load_env()

    statements = json.loads((HERE / "test_statements.json").read_text(encoding="utf-8"))
    cells = plan(statements) + plan_more(statements)
    cells = [c for c in cells if (not a.vendor or c["vendor"] in a.vendor) and (not a.lang or c["language"] in a.lang)]
    if a.one_per_vendor:
        seen = set()
        cells = [c for c in cells if "unsupported" not in c and c["vendor"] not in seen and not seen.add(c["vendor"])]
    runnable = [c for c in cells if "unsupported" not in c]
    print(f"{len(cells)} cells: {len(runnable)} to generate, {len(cells) - len(runnable)} unsupported")
    for v in VENDORS:
        n = sum(c["vendor"] == v for c in runnable)
        print(f"  {v:7} {n:3} clips  key {'set' if os.environ.get(VENDORS[v][0]) else 'MISSING - will skip'}")
    if a.dry_run:
        for c in cells:
            print(f"  {c['vendor']:7} {c['language']:8} {c['accent']:13} {c['voice']:16} {c['variant']:11} "
                  f"{c.get('unsupported') or c['params']}")
        return

    a.out.mkdir(parents=True, exist_ok=True)
    manifest = open(a.out / "manifest.jsonl", "a", encoding="utf-8")

    def run(c):
        rec = {k: c[k] for k in ("vendor", "language", "accent", "voice", "variant")}
        rec["chars"] = len(c["text"])
        rec["support"] = c.get("support", "unsupported" if "unsupported" in c else "documented")
        if "unsupported" in c:
            return {**rec, "status": "unsupported", "detail": c["unsupported"]}
        key_name, fn = VENDORS[c["vendor"]]
        if not os.environ.get(key_name):
            return {**rec, "status": "skipped", "detail": f"{key_name} not set"}
        done = [p for p in out_path(a.out, c, "*").parent.glob(out_path(a.out, c, "x").stem + ".*")]
        if done:
            return {**rec, "status": "exists", "file": str(done[0])}
        try:
            audio, ext, meta = fn(c)
        except Unsupported as exc:
            return {**rec, "status": "unsupported", "support": "unsupported", "detail": str(exc)}
        except Exception as exc:
            return {**rec, "status": "error", "detail": str(exc)[:500]}
        p = out_path(a.out, c, ext)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(audio)
        return {**rec, "status": "ok", "file": str(p), "bytes": len(audio), **meta}

    with ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(run, cells):
            manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
            manifest.flush()
            if rec["status"] not in ("unsupported", "exists"):
                print(f"{rec['status']:8} {rec['vendor']:7} {rec['language']:8} {rec['accent']:13} {rec['voice']:16} "
                      f"{rec['variant']:11} {rec.get('seconds', '')} {rec.get('detail', '')[:120]}", flush=True)
    print(f"manifest: {a.out / 'manifest.jsonl'}")


if __name__ == "__main__":
    sys.exit(main())
