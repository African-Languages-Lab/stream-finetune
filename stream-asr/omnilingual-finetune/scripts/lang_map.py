"""Our 38 languages mapped onto Omnilingual ASR's language-conditioning codes.

Omnilingual conditions the LLM decoder on a language token (see
omnilingual_asr/models/wav2vec2_llama/lang_ids.py, 1668 codes). Getting this mapping right
is what makes one combined model answer in the right language, so each entry below was
checked against that list rather than assumed.

Two traps caught while building it:

  sepedi vs sesotho -- BOTH are Sotho languages and it is tempting to map them to the same
  code. nso_Latn is Northern Sotho, which IS sepedi. Sesotho is sot_*, and that is NOT in
  the supported list. Mapping sesotho to nso_Latn would silently train it as sepedi.

  oromo -- the NLLB-style code gaz_Latn (West Central Oromo) is absent; plain orm_Latn is
  present. Guessing from NLLB conventions would have marked it unsupported.

Every code in LANG_MAP was checked against the model's own lookup table
(omnilingual_asr/models/wav2vec2_llama/languges_lookup_table.parquet, 1,693 rows -- the table
the language embedding is actually built from, larger than lang_ids.py). Two corrections came
out of that check: twi has its own twi_Latn (more specific than the Akan macro-code aka_Latn),
and berber has zgh_Tfng (Standard Moroccan Tamazight in Tifinagh script).

NEW_LANG_CODES holds the four languages the table has no row for. Their codes follow the
table's own convention, ISO 639-3 + ISO 15924 script. They cannot be used by simply naming
them: the language embedding has exactly len(table)+1 = 1,694 rows, so these need four rows
appended to the lookup table and the checkpoint's lang embedding grown to 1,698, with the new
rows initialised from the closest existing language (init_from). Until that is done the model
maps them to id 0 ("unknown language"), which trains but is unconditioned.
"""

# our corpus code -> (plain name, omnilingual language token)
LANG_MAP = {
    "af-ZA":  ("afrikaans",    "afr_Latn"),
    "am-ET":  ("amharic",      "amh_Ethi"),
    "ar-AR":  ("arabic",       "arb_Arab"),
    "bm-ML":  ("bambara",      "bam_Latn"),
    "bem-ZM": ("bemba",        "bem_Latn"),
    "ny-MW":  ("chichewa",     "nya_Latn"),
    # One English: en-GH/NG/UG/ZA and the English moved out of other languages are merged into
    # corpus code "en" by data-prep/build_corpus.py (row "origin" keeps where each came from).
    "en":     ("english",      "eng_Latn"),
    "ee-GH":  ("ewe",          "ewe_Latn"),
    "fon-BJ": ("fon",          "fon_Latn"),
    "ff-SN":  ("fula",         "fuv_Latn"),
    "ha-NG":  ("hausa",        "hau_Latn"),
    "ig-NG":  ("igbo",         "ibo_Latn"),
    "kr-NG":  ("kanuri",       "knc_Latn"),
    "ki-KE":  ("kikuyu",       "kik_Latn"),
    "rw-RW":  ("kinyarwanda",  "kin_Latn"),
    "kri-SL": ("krio",         "kri_Latn"),
    "ln-CD":  ("lingala",      "lin_Latn"),
    "lg-UG":  ("luganda",      "lug_Latn"),
    "mg-MG":  ("malagasy",     "plt_Latn"),
    "or-KE":  ("oromo",        "orm_Latn"),
    "nso-ZA": ("sepedi",       "nso_Latn"),
    "sn-ZW":  ("shona",        "sna_Latn"),
    "so-SO":  ("somali",       "som_Latn"),
    "sw-KE":  ("swahili",      "swh_Latn"),
    "ti-ER":  ("tigrinya",     "tir_Ethi"),
    "ts-ZA":  ("tsonga",       "tso_Latn"),
    "tn-BW":  ("tswana",       "tsn_Latn"),
    "tw-GH":  ("twi",          "twi_Latn"),
    "umb-AO": ("umbundu",      "umb_Latn"),
    "wo-SN":  ("wolof",        "wol_Latn"),
    "xh-ZA":  ("xhosa",        "xho_Latn"),
    "yo-NG":  ("yoruba",       "yor_Latn"),
    "zu-ZA":  ("zulu",         "zul_Latn"),
    # Only 1.4 h of our berber text is actually Tifinagh; ~5 h is Latin script and 28 h is
    # Arabic-script text from Tamazight-Speech-to-Arabic-Text, which is a translation corpus,
    # not Berber transcription. That source should be excluded before training, not relabelled.
    "ber-MA": ("berber",       "zgh_Tfng"),
}

# Not in the lookup table; codes follow its ISO 639-3 + script convention. See module docstring.
NEW_LANG_CODES = {
    "st-ZA":  ("sesotho",  "sot_Latn", "nso_Latn"),   # init from sepedi (Sotho-Tswana group)
    # Our corpus code says Zimbabwe (Northern Ndebele, nde) but every source is South African
    # Ndebele (anv-nbl, nchlt-nbl-*), which is ISO 639-3 nbl.
    "nd-ZW":  ("ndebele",  "nbl_Latn", "zul_Latn"),   # init from zulu (Nguni group)
    "ss-SZ":  ("swati",    "ssw_Latn", "zul_Latn"),   # init from zulu (Nguni group)
    "ve-ZA":  ("venda",    "ven_Latn", "tso_Latn"),   # no close relative; tsonga is the nearest neighbour in our data
}

# Kept for callers that still import it; every language now has a code.
UNSUPPORTED = {}


def token_for(corpus_code: str):
    """Language token for a corpus code, or None if the language has no code yet."""
    entry = LANG_MAP.get(corpus_code) or NEW_LANG_CODES.get(corpus_code)
    return entry[1] if entry else None


def is_new_code(corpus_code: str) -> bool:
    """True when the token has no row in the checkpoint's language table yet."""
    return corpus_code in NEW_LANG_CODES


def name_for(corpus_code: str):
    entry = LANG_MAP.get(corpus_code) or NEW_LANG_CODES.get(corpus_code)
    return entry[0] if entry else corpus_code


if __name__ == "__main__":
    print(f"{len(LANG_MAP)} corpus codes with existing Omnilingual tokens "
          f"({len({v[1] for v in LANG_MAP.values()})} distinct)")
    for code, (name, tok) in sorted(LANG_MAP.items()):
        print(f"  {code:8}{name:14}{tok}")
    print(f"\n{len(NEW_LANG_CODES)} new codes (need lookup-table rows + embedding growth):")
    for code, (name, tok, init) in sorted(NEW_LANG_CODES.items()):
        print(f"  {code:8}{name:14}{tok:10} init from {init}")
