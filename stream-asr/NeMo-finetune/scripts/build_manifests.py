"""
Build NeMo-format training manifests for fine-tuning nemotron-3.5-asr-streaming-0.6b
on ALL available African-language speech data in
/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out,
regardless of the `task` tag (ASR/TTS/ST/NLU/pronunciation/code-switching_ASR/other) --
every row with a non-empty transcript is included.

Design: rather than trying to reconstruct each row's audio folder from the CSV's
`dataset_name` (inconsistent with actual on-disk folder names for ~21 of 38 languages),
we walk each language's audio/ directory ONCE and build a filename-stem -> full-path
index, then look up `audio_id` directly.

English handling, three layers:
1. Bulk country-level dialect tags (Ghanaian_English, Nigerian_English,
   South_African_English, Ugandan_English) get their own standalone prompt codes
   (en-GH/en-NG/en-ZA/en-UG).
2. Rows tagged with a specific L1/ethnicity dialect matching one of our other 38
   languages (e.g. dialect=="yoruba") are merged directly into that language's own
   manifest/prompt slot -- we know precisely which language they belong to.
3. The bulk country-level English (#1) is ALSO duplicated into every local language
   spoken in that country, for code-switching robustness -- uncapped for
   Nigeria/South Africa/Uganda (their English volume is tiny relative to the local
   languages, so duplication is harmless reinforcement), but CAPPED to each language's
   own native hour count for Ghana specifically, where the English volume (2706.6h)
   dwarfs Twi+Ewe combined (341.5h) and uncapped duplication would make those slots
   majority-English rather than majority-Twi/Ewe.

Manifests are keyed by target_lang code (not source directory name) so all of the
above merges naturally into the same per-target_lang files.
"""
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

csv.field_size_limit(sys.maxsize)

SPEECH_OUT = Path("/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out")
OUT_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
random.seed(0)

LANG_CONFIG = {
    "swahili":     ("sw-KE", 48, True),
    "amharic":     ("am-ET", 49, True),
    "hausa":       ("ha-NG", 50, True),
    "zulu":        ("zu-ZA", 51, True),
    "yoruba":      ("yo-NG", 52, True),
    "igbo":        ("ig-NG", 53, True),
    "afrikaans":   ("af-ZA", 54, True),
    "kinyarwanda": ("rw-RW", 55, True),
    "somali":      ("so-SO", 56, True),
    "chichewa":    ("ny-MW", 57, True),
    "lingala":     ("ln-CD", 58, True),
    "oromo":       ("or-KE", 59, True),
    "arabic":      ("ar-AR", 7, True),
    "twi":       ("tw-GH", 63, False),
    "ewe":       ("ee-GH", 72, False),
    "sesotho":   ("st-ZA", 73, False),
    "tswana":    ("tn-BW", 74, False),
    "ndebele":   ("nd-ZW", 75, False),
    "tsonga":    ("ts-ZA", 76, False),
    "fula":      ("ff-SN", 77, False),
    "bambara":   ("bm-ML", 78, False),
    "kikuyu":    ("ki-KE", 79, False),
    "xhosa":     ("xh-ZA", 84, False),
    "venda":     ("ve-ZA", 85, False),
    "tigrinya":  ("ti-ER", 86, False),
    "luganda":   ("lg-UG", 87, False),
    "swati":     ("ss-SZ", 88, False),
    "bemba":     ("bem-ZM", 89, False),
    "sepedi":    ("nso-ZA", 90, False),
    "malagasy":  ("mg-MG", 91, False),
    "shona":     ("sn-ZW", 92, False),
    "kanuri":    ("kr-NG", 93, False),
    "fon":       ("fon-BJ", 94, False),
    "krio":      ("kri-SL", 95, False),
    "berber":    ("ber-MA", 105, False),
    "wolof":     ("wo-SN", 106, False),
    "umbundu":   ("umb-AO", 107, False),
}

ENGLISH_STANDALONE = {
    "Ghanaian_English":      ("en-GH", 108, False),
    "Nigerian_English":      ("en-NG", 109, False),
    "South_African_English": ("en-ZA", 110, False),
    "Ugandan_English":       ("en-UG", 111, False),
}

ENGLISH_MERGE_INTO_LANGUAGE = {
    "yoruba": "yo-NG",
    "igbo": "ig-NG",
    "hausa": "ha-NG",
    "swahili": "sw-KE",
    "kiswahili": "sw-KE",
    "zulu": "zu-ZA",
    "isizulu": "zu-ZA",
    "twi": "tw-GH",
    "akan (fante)": "tw-GH",
    "afrikaans": "af-ZA",
    "setswana": "tn-BW",
    "tswana": "tn-BW",
    "luganda": "lg-UG",
    "kinyarwanda": "rw-RW",
}

# Bulk country-level English duplicated into every local language of that country.
# Ghana is capped (see module docstring); the rest are duplicated uncapped since
# their English volume is negligible relative to the local languages.
COUNTRY_LANGUAGES = {
    "Nigerian_English":      {"langs": ["yo-NG", "ig-NG", "ha-NG"], "cap_to_native": False},
    "South_African_English": {"langs": ["zu-ZA", "xh-ZA", "af-ZA", "st-ZA", "ts-ZA", "ve-ZA", "nso-ZA"], "cap_to_native": False},
    "Ugandan_English":       {"langs": ["lg-UG"], "cap_to_native": False},
    "Ghanaian_English":      {"langs": ["tw-GH", "ee-GH"], "cap_to_native": True},
}


def build_audio_index(lang_dir: Path) -> dict:
    """stem -> absolute path, for every audio file under lang_dir/audio."""
    index = {}
    collisions = 0
    audio_root = lang_dir / "audio"
    if not audio_root.is_dir():
        return index
    for root, _dirs, files in os.walk(audio_root):
        for fname in files:
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in AUDIO_EXTS:
                continue
            if stem in index:
                collisions += 1
                continue
            index[stem] = os.path.join(root, fname)
    if collisions:
        print(f"    [warn] {collisions} duplicate audio_id stems seen, kept first occurrence")
    return index


class ManifestWriterPool:
    """Opens {target_lang}_{split}.jsonl in append mode, lazily, one handle per (target_lang, split)."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.handles = {}

    def get(self, target_lang: str, split: str):
        if split not in ("train", "dev", "test"):
            split = "train"
        key = (target_lang, split)
        if key not in self.handles:
            path = self.out_dir / f"{target_lang}_{split}.jsonl"
            self.handles[key] = open(path, "a", encoding="utf-8")
        return self.handles[key]

    def close_all(self):
        for h in self.handles.values():
            h.close()


def process_language(pool: ManifestWriterPool, lang_dir_name: str, target_lang: str, prompt_index: int, reserved: bool):
    lang_path = SPEECH_OUT / lang_dir_name
    csv_path = lang_path / f"speech_transcribed_{lang_dir_name}.csv"
    if not csv_path.exists():
        print(f"[{lang_dir_name}] SKIP: no csv at {csv_path}")
        return None

    t0 = time.time()
    index = build_audio_index(lang_path)
    t_index = time.time() - t0
    print(f"[{lang_dir_name}] indexed {len(index)} audio files in {t_index:.1f}s")

    stats = {"considered": 0, "written": 0, "missing_audio": 0, "hours_written": 0.0,
             "target_lang": target_lang, "prompt_index": prompt_index, "reserved_slot": reserved,
             "task_types": {}}

    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            text = (row.get("transcript") or "").strip()
            if not text:
                continue
            stats["considered"] += 1
            task = row.get("task") or "unknown"
            stats["task_types"][task] = stats["task_types"].get(task, 0) + 1
            audio_id = row["audio_id"]
            path = index.get(audio_id)
            if path is None:
                stats["missing_audio"] += 1
                continue
            try:
                duration = float(row.get("duration_sec") or 0)
            except ValueError:
                duration = 0.0
            split = row.get("split") or "train"
            rec = {
                "audio_filepath": path,
                "duration": duration,
                "text": text,
                "target_lang": target_lang,
            }
            pool.get(target_lang, split).write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["written"] += 1
            stats["hours_written"] += duration / 3600

    print(f"[{lang_dir_name}] considered={stats['considered']} written={stats['written']} "
          f"missing={stats['missing_audio']} hours={stats['hours_written']:.1f} "
          f"target_lang={target_lang} (prompt_index={prompt_index}, reserved={reserved}) "
          f"tasks={stats['task_types']}")
    return stats


def process_english(pool: ManifestWriterPool, native_hours: dict):
    lang_path = SPEECH_OUT / "english"
    csv_path = lang_path / "speech_transcribed_english.csv"
    t0 = time.time()
    index = build_audio_index(lang_path)
    print(f"[english] indexed {len(index)} audio files in {time.time()-t0:.1f}s")

    all_stats = {}

    def get_stats(bucket_name, target_lang, prompt_index=None, reserved=None):
        if bucket_name not in all_stats:
            all_stats[bucket_name] = {"considered": 0, "written": 0, "missing_audio": 0, "hours_written": 0.0,
                                       "target_lang": target_lang, "prompt_index": prompt_index,
                                       "reserved_slot": reserved, "task_types": {}}
        return all_stats[bucket_name]

    # rows collected per standalone country-dialect, for the duplication pass below.
    standalone_rows = {d: [] for d in ENGLISH_STANDALONE}

    excluded_dialects = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            text = (row.get("transcript") or "").strip()
            if not text:
                continue
            dialect = row.get("dialect") or ""

            if dialect in ENGLISH_STANDALONE:
                target_lang, prompt_index, reserved = ENGLISH_STANDALONE[dialect]
                bucket = f"standalone_{dialect}"
            elif dialect.lower() in ENGLISH_MERGE_INTO_LANGUAGE:
                target_lang = ENGLISH_MERGE_INTO_LANGUAGE[dialect.lower()]
                prompt_index, reserved = None, None
                bucket = f"merged_{dialect.lower()}_into_{target_lang}"
            else:
                excluded_dialects[dialect] = excluded_dialects.get(dialect, 0) + 1
                continue

            stats = get_stats(bucket, target_lang, prompt_index, reserved)
            stats["considered"] += 1
            task = row.get("task") or "unknown"
            stats["task_types"][task] = stats["task_types"].get(task, 0) + 1
            audio_id = row["audio_id"]
            path = index.get(audio_id)
            if path is None:
                stats["missing_audio"] += 1
                continue
            try:
                duration = float(row.get("duration_sec") or 0)
            except ValueError:
                duration = 0.0
            split = row.get("split") or "train"
            rec = {
                "audio_filepath": path,
                "duration": duration,
                "text": text,
                "target_lang": target_lang,
            }
            pool.get(target_lang, split).write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["written"] += 1
            stats["hours_written"] += duration / 3600

            if dialect in standalone_rows:
                standalone_rows[dialect].append((path, duration, text, split))

    print(f"[english] excluded dialects (no matching language slot): "
          f"{sum(excluded_dialects.values())} rows across {len(excluded_dialects)} distinct tags")
    print(f"  top excluded: {sorted(excluded_dialects.items(), key=lambda x: -x[1])[:15]}")
    for bucket, stats in all_stats.items():
        print(f"[english/{bucket}] considered={stats['considered']} written={stats['written']} "
              f"missing={stats['missing_audio']} hours={stats['hours_written']:.1f} "
              f"target_lang={stats['target_lang']} tasks={stats['task_types']}")

    # --- Duplicate bulk country English into the local languages of that country ---
    for dialect, cfg in COUNTRY_LANGUAGES.items():
        rows = standalone_rows[dialect]
        total_hours = sum(d for _, d, _, _ in rows) / 3600
        for target_lang in cfg["langs"]:
            if cfg["cap_to_native"]:
                cap_hours = native_hours.get(target_lang, 0.0)
                if cap_hours <= 0 or total_hours <= 0:
                    chosen = []
                else:
                    frac = min(1.0, cap_hours / total_hours)
                    n_sample = max(1, int(len(rows) * frac))
                    chosen = random.sample(rows, min(n_sample, len(rows)))
            else:
                chosen = rows

            bucket = f"country_dup_{dialect}_into_{target_lang}"
            stats = get_stats(bucket, target_lang)
            for path, duration, text, split in chosen:
                rec = {"audio_filepath": path, "duration": duration, "text": text, "target_lang": target_lang}
                pool.get(target_lang, split).write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["written"] += 1
                stats["considered"] += 1
                stats["hours_written"] += duration / 3600
            print(f"[english/{bucket}] duplicated {len(chosen)} rows ({stats['hours_written']:.1f}h) "
                  f"into {target_lang} (source pool: {len(rows)} rows / {total_hours:.1f}h"
                  f"{', capped to native ' + format(native_hours.get(target_lang, 0.0), '.1f') + 'h' if cfg['cap_to_native'] else ', uncapped'})")

    return all_stats


def main():
    for f in OUT_DIR.glob("*.jsonl"):
        f.unlink()

    pool = ManifestWriterPool(OUT_DIR)
    summary = {}
    try:
        native_hours = {}
        for lang_dir_name, (target_lang, prompt_index, reserved) in LANG_CONFIG.items():
            stats = process_language(pool, lang_dir_name, target_lang, prompt_index, reserved)
            if stats:
                summary[lang_dir_name] = stats
                native_hours[target_lang] = stats["hours_written"]

        english_stats = process_english(pool, native_hours)
        for bucket, stats in english_stats.items():
            summary[f"english_{bucket}"] = stats
    finally:
        pool.close_all()

    with open(OUT_DIR / "manifest_build_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    total_hours = sum(s["hours_written"] for s in summary.values())
    total_written = sum(s["written"] for s in summary.values())
    print(f"\n=== DONE: {total_written} utterances, {total_hours:.1f} hours written across {len(summary)} source buckets ===")


if __name__ == "__main__":
    main()
