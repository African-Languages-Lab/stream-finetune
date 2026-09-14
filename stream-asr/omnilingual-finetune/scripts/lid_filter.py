"""Find the English hiding inside our non-English languages, and write it down.

English leaked into the corpus two different ways, and both had to be handled:

  Whole datasets filed under the wrong language. ghana-english-asr-2700hrs sits in ewe
  (99.5 h) and twi (237.4 h); afrispeech-200 -- which is African-accented ENGLISH -- sits
  in ten languages; the Nigerian English crowdsourced set sits in hausa, igbo and yoruba.

  Individual English utterances inside otherwise-native datasets. The NCHLT aux sets are
  the worst: 10-25 % of nchlt-sot-aux1, nchlt-ssw-aux1 and nchlt-tsn-aux1 are English
  phrases ("united states senate", "central processing unit") read by Sotho/Swati/Tswana
  speakers and filed under the Bantu language.

A source-name blocklist only catches the first kind, so this classifies every transcript
with fastText LID (facebook/fasttext-language-identification, the NLLB-200 218-label
model) and drops the rows it calls English.

WHY THIS DOES NOT EAT THE NATIVE DATA. Measured before trusting it: native sources come
back at 0.0-0.2 % English (Waxal ewe 0.0, Naija_Voices_Igbo 0.2, akuapem-twi 0.0) while
the planted ones come back at 85-100 %. The separation is not marginal.

A SOURCE THAT IS MOSTLY ENGLISH IS DROPPED WHOLE. afrispeech-200 is African-accented
English; the one row in it the code-switch test wanted to keep was "Plan to check Lytes
Ifechukwukwuru BUN/Cr CBC" -- English carrying an Igbo name, which is not an Igbo speaker
code-switching. So once a source under a non-English language comes back over WHOLESALE
English, the whole source goes, code-switch exemption included. The exemption is for English
appearing inside native data, not for native words appearing inside English data.

CODE-SWITCHING IS KEPT, MONOLINGUAL ENGLISH IS NOT. A Yoruba speaker saying "Mo ti book
appointment fun Monday morning" is speaking Yoruba; a recording of "united states senate"
filed under Sotho is not speaking Sotho. Top-1 English cannot separate those -- the Yoruba
sentence comes back eng_Latn at p=0.92. What separates them is whether the language's OWN
token survives anywhere in the top-5: yor_Latn sits at 0.06 behind the Yoruba sentence,
while "united states senate" has no sot_Latn at any rank and "central processing unit" has
no tsn_Latn. So a row is only called English when English wins AND the native language has
been pushed out of the top-5 entirely (below NATIVE_FLOOR).

This matters because Omnilingual has no mixed-language mode. create_lang_inputs() prepends
exactly one language id per utterance, so the model cannot be told an utterance switches --
it can only learn to switch from data. Filter out the code-switched utterances and a model
conditioned on hau_Latn will never have seen an English word in Hausa context.

KRIO IS THE TRAP, AND A THRESHOLD DOES NOT FIX IT. Krio is an English-lexified creole and
the model has NO kri_Latn label, so it cannot tell Krio from English on principle. A first
pass at 0.90 still deleted "A dɔn was mi fut," and "Aya, a wish in lɛf an de ɔnda mi ed" --
both plain Krio, both called English above p=0.98. No threshold separates a language the
model was never taught.

So for any language whose token is absent from the label set, per-row judgment is switched
off entirely and only the source-level rule applies. An aggregate rate over a whole dataset
stays meaningful when individual verdicts do not, so a wholesale-English dataset misfiled
under Krio would still be caught, while no individual Krio utterance is ever deleted on a
judgment the model is incapable of making. Same for venda and ndebele.

LID_ALIAS handles the near-miss case: our token for Oromo is orm_Latn and the model's label
is gaz_Latn (West Central Oromo). Without the alias Oromo would fall into the no-label path
and lose the native-token test for no reason.

Output is a decision file per language, not a rewritten manifest -- build_manifests.py
reads it and skips those rows. Keeping the decision separate means you can read exactly
what was removed and why, and re-run the build without re-running the model.

    python lid_filter.py --out <dir>                 # all non-English languages
    python lid_filter.py --out <dir> --langs tw-GH,ee-GH
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lang_map import LANG_MAP, NEW_LANG_CODES, token_for

NEMO = "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/manifests"
MODEL = ("/leonardo_scratch/large/userexternal/atsado00/hf_home/hub/"
         "models--facebook--fasttext-language-identification/snapshots/"
         "3af127d4124fc58b75666f3594bb5143b9757e78/model.bin")

ENG = "eng_Latn"
DEFAULT_PROB = 0.50

# Our token -> the LID model's label for the same language, where the two spellings differ.
# nbl and ven have no fastText label; calibration showed the model reads them as zul_Latn
# (48-57 %) and tso_Latn (50 %), so those stand in for the native-token test.
LID_ALIAS = {"orm_Latn": "gaz_Latn", "zgh_Tfng": "tzm_Tfng", "nbl_Latn": "zul_Latn", "ven_Latn": "tso_Latn"}
NATIVE_FLOOR = 0.01   # native token at or above this anywhere in the top-5 => code-switched
WHOLESALE = 0.60      # a source this English under a non-English language is misfiled whole
TOPK = 5

_ws = re.compile(r"\s+")


def normalise(text):
    return _ws.sub(" ", unicodedata.normalize("NFC", text)).strip()


def source_of(path):
    parts = path.split("/speech_out/")[-1].split("/")
    return parts[2] if len(parts) > 2 else "?"


def load_model():
    import fasttext
    return fasttext.load_model(MODEL)


def scan(model, code, split, threshold, native_tok, keep_code_switched):
    """Rows judged English, plus the code-switched rows deliberately kept."""
    src = f"{NEMO}/{code}_{split}.jsonl"
    if not os.path.exists(src):
        return [], [], Counter(), Counter(), Counter()
    dropped, switched = [], []
    seen_h, drop_h, switch_h = Counter(), Counter(), Counter()
    with open(src, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = normalise(r.get("text") or "")
            path = r.get("audio_filepath") or ""
            if not text or not path:
                continue
            dur = float(r.get("duration") or 0)
            s = source_of(path)
            seen_h[s] += dur
            lab, prob = model.predict(text, k=TOPK)
            top = [(l.replace("__label__", ""), float(pp)) for l, pp in zip(lab, prob)]
            lang, p = top[0]
            if lang != ENG or p < threshold:
                continue
            native_p = next((pp for l, pp in top if l == native_tok), 0.0)
            if keep_code_switched and native_p >= NATIVE_FLOOR:
                # English wins the vote but the native language is still in the running:
                # a speaker code-switching, not a misfiled English recording.
                switched.append((path, round(p, 3), round(native_p, 3), s))
                switch_h[s] += dur
                continue
            dropped.append((path, round(p, 3), round(native_p, 3), s))
            drop_h[s] += dur
    return dropped, switched, seen_h, drop_h, switch_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="")
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--eng-prob", type=float, default=DEFAULT_PROB)
    ap.add_argument("--drop-code-switched", action="store_true",
                    help="also drop utterances where the speaker code-switches into English. "
                         "Off by default: Omnilingual has no mixed-language mode, so a model "
                         "only learns to handle code-switching from examples of it.")
    a = ap.parse_args()

    model = load_model()
    labels = {l.replace("__label__", "") for l in model.labels}

    langs = [c.strip() for c in a.langs.split(",") if c.strip()] or sorted({**LANG_MAP, **NEW_LANG_CODES})
    # English is a target language in its own right; only hunt for it inside the others.
    langs = [c for c in langs if token_for(c) != ENG]

    os.makedirs(a.out, exist_ok=True)
    report = []
    for code in langs:
        tok = token_for(code)
        lid_tok = LID_ALIAS.get(tok, tok)
        # No label for this language => the model cannot judge its rows at all. Source-level
        # decisions only; see the Krio note at the top.
        source_only = lid_tok not in labels
        thr = a.eng_prob
        keep_cs = not a.drop_code_switched
        all_dropped, all_switched = [], []
        seen_h, drop_h, switch_h = Counter(), Counter(), Counter()
        for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
            d, sw, s_h, d_h, sw_h = scan(model, code, split, thr, lid_tok, keep_cs)
            all_dropped += d
            all_switched += sw
            seen_h.update(s_h)
            drop_h.update(d_h)
            switch_h.update(sw_h)
        if source_only:
            # Park every per-row verdict; only the wholesale-source rule below may act.
            all_switched += [(path, pp, np_, src) for path, pp, np_, src in all_dropped]
            for src in list(drop_h):
                switch_h[src] += drop_h.pop(src)
            all_dropped = []

        # Promote: in a source that is overwhelmingly English, the code-switch exemption is
        # measuring names inside English, not English inside native speech.
        eng_rate = {src: (drop_h[src] + switch_h[src]) / seen_h[src]
                    for src in seen_h if seen_h[src] > 0}
        wholesale = {src for src, rate in eng_rate.items() if rate >= WHOLESALE}
        if wholesale:
            keep = []
            for row in all_switched:
                (keep, all_dropped)[row[3] in wholesale].append(row)
            all_switched = keep
            for src in wholesale:
                drop_h[src] += switch_h.pop(src, 0.0)

        # Every decision is final by this point -- files are written last on purpose, so the
        # report and the files can never disagree.
        header = (f"# {code} token={tok} lid_token={lid_tok} threshold={thr} "
                  f"native_floor={NATIVE_FLOOR}"
                  f"{'  SOURCE-LEVEL ONLY: the LID model has no label for this language, '
                     'so no per-row deletion' if source_only else ''}")
        with open(f"{a.out}/{code}.english.tsv", "w", encoding="utf-8") as fh:
            fh.write(header + "\n# path\teng_p\tnative_p\tsource\n")
            for path, pp, np_, src in all_dropped:
                fh.write(f"{path}\t{pp}\t{np_}\t{src}\n")
        # Written out too, so "what did we keep and why" is as auditable as what we dropped.
        with open(f"{a.out}/{code}.codeswitched.tsv", "w", encoding="utf-8") as fh:
            why = ("not judged: no LID label for this language, so these rows are left in"
                   if source_only else
                   f"kept: English won but the native token held >= {NATIVE_FLOOR} "
                   f"in the top-{TOPK}")
            fh.write(header + f"\n# {why}\n")
            for path, pp, np_, src in all_switched:
                fh.write(f"{path}\t{pp}\t{np_}\t{src}\n")

        tot_seen = sum(seen_h.values()) / 3600
        tot_drop = sum(drop_h.values()) / 3600
        tot_sw = sum(switch_h.values()) / 3600
        pct = 100 * tot_drop / tot_seen if tot_seen else 0
        report.append((code, tok, thr, len(all_dropped), len(all_switched),
                       tot_seen, tot_drop, tot_sw, pct,
                       sorted(((drop_h[s] / 3600, s, seen_h[s] / 3600)
                               for s in drop_h if drop_h[s] > 0), reverse=True)[:3]))

    report.sort(key=lambda r: -r[6])
    print(f"{'lang':9}{'token':10}{'thr':>5}{'dropped':>10}{'hours in':>10}"
          f"{'eng h':>8}{'%':>7}{'cs kept':>9}   biggest offenders")
    for code, tok, thr, n, nsw, seen, drop, sw, pct, top in report:
        off = "; ".join(f"{s} {h:.0f}/{tot:.0f}h" for h, s, tot in top)
        print(f"{code:9}{tok:10}{thr:>5.2f}{n:>10,}{seen:>10.1f}{drop:>8.1f}"
              f"{pct:>6.1f}%{sw:>8.1f}h   {off}")
    print(f"\nEnglish removed from non-English languages: "
          f"{sum(r[6] for r in report):.1f} h across {sum(r[3] for r in report):,} utterances")
    print(f"code-switched speech kept: {sum(r[7] for r in report):.1f} h across "
          f"{sum(r[4] for r in report):,} utterances")
    print(f"decisions written to {a.out}/<lang>.english.tsv (+ .codeswitched.tsv)")


if __name__ == "__main__":
    main()
