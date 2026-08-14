import yaml
from pathlib import Path

# Original 90 unique indices from the real checkpoint (verified by loading the .nemo directly).
EXISTING = {
    'en-US': 0, 'en-GB': 1, 'es-ES': 2, 'es-US': 3, 'zh-CN': 4, 'zh-TW': 5, 'hi-IN': 6, 'ar-AR': 7,
    'fr-FR': 8, 'de-DE': 9, 'ja-JP': 10, 'ru-RU': 11, 'pt-BR': 12, 'pt-PT': 13, 'ko-KR': 14, 'it-IT': 15,
    'nl-NL': 16, 'pl-PL': 17, 'tr-TR': 18, 'uk-UA': 19, 'ro-RO': 20, 'el-GR': 21, 'cs-CZ': 22, 'hu-HU': 23,
    'sv-SE': 24, 'da-DK': 25, 'fi-FI': 26, 'no-NO': 27, 'sk-SK': 28, 'hr-HR': 29, 'bg-BG': 30, 'lt-LT': 31,
    'th-TH': 32, 'vi-VN': 33, 'id-ID': 34, 'ms-MY': 35, 'bn-IN': 36, 'ur-PK': 37, 'fa-IR': 38, 'ta-IN': 39,
    'te-IN': 40, 'mr-IN': 41, 'gu-IN': 42, 'kn-IN': 43, 'ml-IN': 44, 'si-LK': 45, 'ne-NP': 46, 'km-KH': 47,
    'sw-KE': 48, 'am-ET': 49, 'ha-NG': 50, 'zu-ZA': 51, 'yo-NG': 52, 'ig-NG': 53, 'af-ZA': 54, 'rw-RW': 55,
    'so-SO': 56, 'ny-MW': 57, 'ln-CD': 58, 'or-KE': 59, 'et-EE': 60, 'lv-LV': 61, 'sl-SI': 62, 'he-IL': 64,
    'ku-TR': 65, 'az-AZ': 66, 'ka-GE': 67, 'hy-AM': 68, 'uz-UZ': 69, 'tg-TJ': 70, 'ky-KG': 71, 'qu-PE': 80,
    'ay-BO': 81, 'gn-PY': 82, 'nah-MX': 83, 'mi-NZ': 96, 'haw-US': 97, 'sm-WS': 98, 'to-TO': 99,
    'fr-CA': 100, 'auto': 101, 'mt-MT': 102, 'nb-NO': 103, 'nn-NO': 104,
}

NEW = {
    'tw-GH': 63, 'ee-GH': 72, 'st-ZA': 73, 'tn-BW': 74, 'nd-ZW': 75, 'ts-ZA': 76, 'ff-SN': 77, 'bm-ML': 78,
    'ki-KE': 79, 'xh-ZA': 84, 've-ZA': 85, 'ti-ER': 86, 'lg-UG': 87, 'ss-SZ': 88, 'bem-ZM': 89, 'nso-ZA': 90,
    'mg-MG': 91, 'sn-ZW': 92, 'kr-NG': 93, 'fon-BJ': 94, 'kri-SL': 95, 'ber-MA': 105, 'wo-SN': 106,
    'umb-AO': 107, 'en-GH': 108, 'en-NG': 109, 'en-ZA': 110, 'en-UG': 111,
}

merged = {**EXISTING, **NEW}
assert len(set(merged.values())) <= 128

OUT_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/configs")
with open(OUT_DIR / "prompt_dictionary.yaml", "w") as f:
    f.write("prompt_dictionary:\n")
    for k, v in merged.items():
        f.write(f"  {k}: {v}\n")

print(f"Total codes: {len(merged)}, unique indices: {len(set(merged.values()))}, max index: {max(merged.values())}")
