"""ElevenLabs, Spitch, Khaya, Abena AI, 9jaLingo, Google Cloud and Azure adapters.

Each vendor declares what it covers. A (language, accent, gender) the vendor does not offer
becomes an "unsupported" cell with the reason, so the comparison shows the gap. Google and
Azure publish long, changing voice lists, so their voices are chosen at run time from the
live voice list; a locale with no voice is recorded as unsupported then.

Keys (stream/.env):
    ELEVENLABS_API_KEY  SPITCH_API_KEY  KHAYA_API_KEY  ABENA_API_KEY  NAIJALINGO_API_KEY
    GOOGLE_TTS_API_KEY                  AZURE_SPEECH_KEY + AZURE_SPEECH_REGION
"""
import base64
import html
import io
import os
import time
import wave

import requests

GENDERS = ("male", "female")
ACCENTS = ("nigerian", "ghanaian", "east_african")


class Unsupported(Exception):
    """Raised at run time when a live voice list has nothing for the request."""


def _post(url, retries=4, **kw):
    for attempt in range(retries):
        r = requests.post(url, timeout=kw.pop("timeout", 600), **kw)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(float(r.headers.get("Retry-After") or 20 * (attempt + 1)))
            continue
        return r
    return r


def _pcm_to_wav(pcm, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate), w.writeframes(pcm)
    return buf.getvalue()


# --------------------------------------------------------------------------- ElevenLabs
# eleven_v3 is the only ElevenLabs model with Hausa and Swahili (checked via GET /v1/models).
# Accent voices come from the shared voice library (GET /v1/shared-voices). Nothing there is
# labelled Ghanaian, so Ghanaian English is unsupported. The Kenyan voices are labelled
# "received pronunciation" but are named and described as Kenyan speakers.
EL_VOICES = {
    ("english", "nigerian", "female"): ("JMwQvjJt08OhYlPBWeyc", "Tolani"),
    ("english", "nigerian", "male"): ("TyAD2ntJFdDReoa55SLn", "Bill"),
    ("english", "east_african", "female"): ("ijKilL5CnjXKMWDHOJH8", "Yna Agalo"),
    ("english", "east_african", "male"): ("uLfPT2jUO3X81OwnftBP", "Keith Muoki"),
    ("hausa", "", "female"): ("JMwQvjJt08OhYlPBWeyc", "Tolani"),   # no native Hausa voice exists
    ("hausa", "", "male"): ("TyAD2ntJFdDReoa55SLn", "Bill"),
    ("swahili", "", "female"): ("ijKilL5CnjXKMWDHOJH8", "Yna Agalo"),  # no native Swahili voice exists
    ("swahili", "", "male"): ("uLfPT2jUO3X81OwnftBP", "Keith Muoki"),
}
EL_LANG = {"english": "en", "hausa": "ha", "swahili": "sw"}
_el_added = set()


def _el_ensure_voice(voice_id, name, key):
    """Shared voices must be in the account's library before they can be used."""
    if voice_id in _el_added:
        return
    h = {"xi-api-key": key}
    if requests.get(f"https://api.elevenlabs.io/v1/voices/{voice_id}", headers=h, timeout=60).status_code == 200:
        _el_added.add(voice_id)
        return
    found = requests.get("https://api.elevenlabs.io/v1/shared-voices", headers=h, timeout=60,
                         params={"search": name, "page_size": 100}).json().get("voices", [])
    match = next((v for v in found if v.get("voice_id") == voice_id), None)
    if not match:
        raise RuntimeError(f"shared voice {name} ({voice_id}) not found in the library")
    r = requests.post(f"https://api.elevenlabs.io/v1/voices/add/{match['public_owner_id']}/{voice_id}",
                      headers=h, json={"new_name": f"bench {name}"}, timeout=60)
    if r.status_code not in (200, 201) and "already" not in r.text.lower():
        raise RuntimeError(f"could not add shared voice {name}: HTTP {r.status_code} {r.text[:200]}")
    _el_added.add(voice_id)


def elevenlabs(c):
    key = os.environ["ELEVENLABS_API_KEY"]
    voice_id, name = c["params"]["voice_id"], c["params"]["voice_name"]
    _el_ensure_voice(voice_id, name, key)
    t0 = time.time()
    r = _post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128",
              headers={"xi-api-key": key},
              json={"text": c["text"], "model_id": "eleven_v3", "language_code": c["params"]["language_code"]})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.content, "mp3", {"seconds": round(time.time() - t0, 2), "voice_name": name}


# --------------------------------------------------------------------------- Spitch
# docs.spitch.app/concepts/voices. English voices are from a Nigerian company; no accent is
# stated, so they are filed under Nigerian English only.
SPITCH_VOICES = {"english": ("john", "lucy"), "hausa": ("aliyu", "amina"),
                 "igbo": ("obinna", "ngozi"), "yoruba": ("femi", "sade")}
SPITCH_LANG = {"english": "en", "hausa": "ha", "igbo": "ig", "yoruba": "yo"}


def spitch(c):
    t0 = time.time()
    r = _post("https://api.spitch.app/v1/speech",
              headers={"Authorization": f"Bearer {os.environ['SPITCH_API_KEY']}"},
              json={"text": c["text"], "language": c["params"]["language"], "voice": c["params"]["voice"],
                    "format": "wav"})
    if r.status_code != 200 or not r.content.startswith(b"RIFF"):
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.content, "wav", {"seconds": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------- Khaya (GhanaNLP)
# Endpoint, header and speaker ids from the official khaya-sdk source. Asante Twi ("twi")
# is used for Twi. English ("eng") is a Ghanaian product, filed under Ghanaian English.
KHAYA_LANG = {"twi": "twi", "ewe": "ewe", "hausa": "hau", "igbo": "ibo", "yoruba": "yor",
              "swahili": "swa", "english": "eng"}
KHAYA_SPEAKER = {"male": "male_low", "female": "female"}


def khaya(c):
    t0 = time.time()
    r = _post(os.environ.get("KHAYA_BASE_URL", "https://translation-api.ghananlp.org") + "/tts/v1/tts",
              headers={"Ocp-Apim-Subscription-Key": os.environ["KHAYA_API_KEY"]},
              json={"text": c["text"], "language": c["params"]["language"], "speaker_id": c["params"]["speaker_id"]})
    if r.status_code != 200 or not (r.headers.get("content-type", "").startswith("audio/") or r.content.startswith(b"RIFF")):
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.content, "wav" if r.content.startswith(b"RIFF") else "bin", {"seconds": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------- Abena AI
# abena.mobobi.com/playground/sdk/docs. One voice per language, so gender follows the voice.
ABENA_VOICES = {
    ("english", "ghanaian", "female"): "akua_eng", ("english", "ghanaian", "male"): "kwabena_eng",
    ("english", "nigerian", "female"): "chioma_eng",
    ("twi", "", "female"): "abena_twi_high", ("ewe", "", "male"): "mawuli_ewe",
    ("hausa", "", "male"): "abubakar_hau", ("yoruba", "", "female"): "folami_yor",
    ("swahili", "", "unspecified"): "amani_swh",
}


def abena(c):
    t0 = time.time()
    r = _post("https://abena.mobobi.com/playground/api/v1/tts/synthesize/",
              headers={"Authorization": f"Bearer {os.environ['ABENA_API_KEY']}"},
              json={"text": c["text"], "voice": c["params"]["voice"], "speed": 1.0})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    if j.get("status") != "success" or not j.get("audio_base64"):
        raise RuntimeError(f"no audio: {str(j)[:300]}")
    return base64.b64decode(j["audio_base64"]), "wav", {"seconds": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------- 9jaLingo
# Base URL, header and routes from the official naijalingo SDK (2.0.6). Speakers are picked
# from GET /v1/speakers so the ids are real, not guessed.
NAIJA_LANG = {"hausa": "ha", "igbo": "ig", "yoruba": "yo"}
_naija_speakers = {}


def _naija_headers():
    return {"X-API-Key": os.environ["NAIJALINGO_API_KEY"]}


def _naija_speaker(lang, gender):
    if (lang, gender) not in _naija_speakers:
        r = requests.get("https://api.9jalingo.org/v1/speakers", headers=_naija_headers(), timeout=60,
                         params={"language": lang, "gender": gender})
        r.raise_for_status()
        j = r.json()
        items = j.get("speakers") or j.get("data") or (j if isinstance(j, list) else [])
        ids = [s.get("speaker_id") or s.get("id") for s in items if isinstance(s, dict)]
        _naija_speakers[(lang, gender)] = next((i for i in ids if i), None)
    sid = _naija_speakers[(lang, gender)]
    if not sid:
        raise Unsupported(f"9jaLingo lists no {gender} speaker for {lang}")
    return sid


def naijalingo(c):
    lang, gender = c["params"]["lang"], c["params"]["gender"]
    speaker = _naija_speaker(lang, gender)
    t0 = time.time()
    r = _post("https://api.9jalingo.org/v1/audio/speech", headers={**_naija_headers(), "Prefer": "respond-async-on-cold-start"},
              json={"input": c["text"], "voice": speaker, "lang": lang, "response_format": "wav"})
    if r.status_code == 202:
        job = r.json()["job_id"]
        while True:
            time.sleep(3)
            s = requests.get(f"https://api.9jalingo.org/v1/jobs/{job}", headers=_naija_headers(), timeout=60).json()
            if s.get("status") in ("completed", "succeeded", "done", "ready"):
                r = requests.get(f"https://api.9jalingo.org/v1/jobs/{job}/audio", headers=_naija_headers(), timeout=300)
                break
            if s.get("status") in ("failed", "error") or time.time() - t0 > 900:
                raise RuntimeError(f"job {job}: {str(s)[:300]}")
    if r.status_code != 200 or not r.content.startswith(b"RIFF"):
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.content, "wav", {"seconds": round(time.time() - t0, 2), "speaker": speaker}


# --------------------------------------------------------------------------- Google / Azure
LOCALES = {("english", "nigerian"): "en-NG", ("english", "ghanaian"): "en-GH", ("english", "east_african"): "en-KE",
           ("hausa", ""): "ha-NG", ("igbo", ""): "ig-NG", ("yoruba", ""): "yo-NG", ("twi", ""): "ak-GH",
           ("ewe", ""): "ee-GH", ("swahili", ""): "sw-KE"}
_google_voices = None


def google(c):
    global _google_voices
    key = os.environ["GOOGLE_TTS_API_KEY"]
    if _google_voices is None:
        r = requests.get("https://texttospeech.googleapis.com/v1/voices", params={"key": key}, timeout=60)
        r.raise_for_status()
        _google_voices = r.json().get("voices", [])
    locale, gender = c["params"]["locale"], c["params"]["gender"].upper()
    cands = [v for v in _google_voices if locale in v.get("languageCodes", []) and v.get("ssmlGender") == gender]
    if not cands:
        raise Unsupported(f"Google lists no {gender.lower()} voice for {locale}")
    rank = ("Chirp3-HD", "Chirp-HD", "Neural2", "Wavenet", "Standard")
    cands.sort(key=lambda v: next((i for i, t in enumerate(rank) if t in v["name"]), len(rank)))
    voice = cands[0]["name"]
    t0 = time.time()
    r = _post("https://texttospeech.googleapis.com/v1/text:synthesize", params={"key": key},
              json={"input": {"text": c["text"]}, "voice": {"languageCode": locale, "name": voice},
                    "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    audio = base64.b64decode(r.json()["audioContent"])
    return (audio if audio.startswith(b"RIFF") else _pcm_to_wav(audio, 24000)), "wav", \
        {"seconds": round(time.time() - t0, 2), "voice_name": voice}


_azure_voices = None


def azure(c):
    global _azure_voices
    key, region = os.environ["AZURE_SPEECH_KEY"], os.environ.get("AZURE_SPEECH_REGION", "")
    if not region:
        raise RuntimeError("AZURE_SPEECH_REGION not set")
    h = {"Ocp-Apim-Subscription-Key": key}
    if _azure_voices is None:
        r = requests.get(f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list", headers=h, timeout=60)
        r.raise_for_status()
        _azure_voices = r.json()
    locale, gender = c["params"]["locale"], c["params"]["gender"].capitalize()
    cands = [v for v in _azure_voices if v.get("Locale") == locale and v.get("Gender") == gender]
    if not cands:
        raise Unsupported(f"Azure lists no {gender.lower()} voice for {locale}")
    voice = cands[0]["ShortName"]
    ssml = (f"<speak version='1.0' xml:lang='{locale}'><voice name='{voice}'>"
            f"{html.escape(c['text'])}</voice></speak>")
    t0 = time.time()
    r = _post(f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
              headers={**h, "Content-Type": "application/ssml+xml",
                       "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm", "User-Agent": "vendor-benchmark"},
              data=ssml.encode("utf-8"))
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.content, "wav", {"seconds": round(time.time() - t0, 2), "voice_name": voice}


# --------------------------------------------------------------------------- plan
MORE_VENDORS = {
    "elevenlabs": ("ELEVENLABS_API_KEY", elevenlabs), "spitch": ("SPITCH_API_KEY", spitch),
    "khaya": ("KHAYA_API_KEY", khaya), "abena": ("ABENA_API_KEY", abena),
    "naijalingo": ("NAIJALINGO_API_KEY", naijalingo), "google": ("GOOGLE_TTS_API_KEY", google),
    "azure": ("AZURE_SPEECH_KEY", azure),
}


def _slots(lang):
    """(accent, gender) slots for a language: accents for English, none otherwise."""
    accents = ACCENTS if lang == "english" else ("",)
    return [(a, g) for a in accents for g in GENDERS]


def plan_more(statements):
    cells = []
    for lang, texts in statements.items():
        for variant, text in texts.items():
            base = {"language": lang, "variant": variant, "text": text}

            def add(vendor, accent, voice, params=None, why=None):
                cell = {**base, "vendor": vendor, "accent": accent, "voice": voice}
                cell.update({"params": params} if why is None else {"unsupported": why})
                cells.append(cell)

            for accent, g in _slots(lang):
                # ElevenLabs
                v = EL_VOICES.get((lang, accent, g))
                if v:
                    add("elevenlabs", accent, f"{v[1].split()[0].lower()}_{g}",
                        {"voice_id": v[0], "voice_name": v[1], "language_code": EL_LANG[lang]})
                else:
                    add("elevenlabs", accent, g, why=("no Ghanaian English voice in the library" if accent == "ghanaian"
                                                       else f"eleven_v3 does not support {lang}"))
                # Spitch
                if lang in SPITCH_VOICES and accent in ("", "nigerian"):
                    add("spitch", accent, f"{SPITCH_VOICES[lang][GENDERS.index(g)]}_{g}",
                        {"language": SPITCH_LANG[lang], "voice": SPITCH_VOICES[lang][GENDERS.index(g)]})
                else:
                    add("spitch", accent, g, why=f"Spitch offers no {accent + ' ' if accent else ''}{lang}")
                # Khaya
                if lang in KHAYA_LANG and accent in ("", "ghanaian"):
                    add("khaya", accent, g, {"language": KHAYA_LANG[lang], "speaker_id": KHAYA_SPEAKER[g]})
                else:
                    add("khaya", accent, g, why=f"Khaya English is Ghanaian; no {accent} English")
                # 9jaLingo
                if lang in NAIJA_LANG:
                    add("naijalingo", accent, g, {"lang": NAIJA_LANG[lang], "gender": g})
                else:
                    add("naijalingo", accent, g, why=f"9jaLingo offers only Hausa, Igbo, Yoruba and Pidgin, not {lang}")
                # Google and Azure: resolved against the live voice list at run time
                for vendor in ("google", "azure"):
                    add(vendor, accent, g, {"locale": LOCALES[(lang, accent)], "gender": g})
            # Abena: one fixed voice per language
            for (l, accent, g), voice in ABENA_VOICES.items():
                if l == lang:
                    add("abena", accent, f"{voice}_{g}", {"voice": voice})
            if lang not in {k[0] for k in ABENA_VOICES}:
                add("abena", "", "-", why=f"Abena offers no {lang} voice")
            if lang == "english":
                add("abena", "east_african", "-", why="Abena offers no East African English voice")
    return cells
