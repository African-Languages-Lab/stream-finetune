"""Pick the serving checkpoint per language: the LATEST one that produces good audio.

Not the best-scoring one. The score here is a duration check -- it catches a collapse or a
spoken reference, and it is blind to anything finer. So a marginal scoring difference between
two working checkpoints is noise, and breaking the tie by score would hand serving to an
early, under-trained epoch over a later one that had seen far more data. Among everything
that WORKS, the latest wins; the score only decides what counts as working at all.

Epochs are compared NUMERICALLY. Sorting the filenames as text put epoch_130 before epoch_65
and epoch_100 before epoch_16, which silently picked earlier checkpoints for fon and others.

Reads the sweep logs rather than results.json, because successive sweeps overwrote that file.
"""
import json
import os
import re
import sys
from collections import defaultdict

LOGS = "/leonardo_scratch/large/userexternal/atsado00/unserved_sweep/logs"
YORUBA = "/leonardo_scratch/large/userexternal/atsado00/yoruba_sweep/results.json"

HDR = re.compile(r"^=== (?P<lang>[\w_]+) \((?P<iso>[\w-]+)\) -- llm \[(?P<llms>.*?)\] \+ flow (?P<flow>\S+) ===")
ROW = re.compile(r"^(?P<llm>epoch_\S*?)\s*(?P<ref>male|female)\s+\[(?P<zs>[^\]]*)\]\s*(?P<zsv>\w+)\s+\[(?P<xl>[^\]]*)\]\s*(?P<xlv>\w+)\s*$")


def epoch_key(name):
    """(epoch, step) as integers; a whole-epoch checkpoint sorts after its own steps."""
    nums = [int(x) for x in re.findall(r"\d+", name)]
    if not nums:
        return (-1, -1)
    return (nums[0], nums[1] if len(nums) > 1 else 10 ** 9)


def parse_logs():
    rows = defaultdict(list)
    for fn in sorted(os.listdir(LOGS)):
        if not fn.endswith(".out"):
            continue
        lang = flow = None
        for line in open(os.path.join(LOGS, fn), errors="ignore"):
            line = line.rstrip("\n")
            m = HDR.match(line.strip())
            if m:
                lang, flow = m.group("lang"), m.group("flow")
                continue
            if not lang:
                continue
            r = ROW.match(line.strip())
            if r:
                rows[lang].append({
                    "llm": r.group("llm").rstrip("."), "flow": flow, "ref": r.group("ref"),
                    "zs": r.group("zsv"), "xl": r.group("xlv"),
                })
    return rows


def pick(rs):
    """Latest checkpoint whose BOTH genders work, in a single mode. Falls back to one gender."""
    by = defaultdict(dict)
    for r in rs:
        by[r["llm"]].setdefault(r["ref"], {})["zs"] = r["zs"]
        by[r["llm"]][r["ref"]]["xl"] = r["xl"]
    both, single = [], []
    for llm in sorted(by, key=epoch_key):
        for mode in ("zs", "xl"):
            gs = {g: v[mode] for g, v in by[llm].items()}
            ok = [g for g, v in gs.items() if v == "ok"]
            if len(ok) == 2:
                both.append((llm, mode, "both"))
            elif ok:
                single.append((llm, mode, f"{ok[0]} only"))
    pool = both or single
    if not pool:
        return None
    # sorted() above is already numeric-ascending, so the last entry is the latest checkpoint
    return max(pool, key=lambda t: epoch_key(t[0]))


def main():
    rows = parse_logs()
    if os.path.exists(YORUBA):
        for r in json.load(open(YORUBA)):
            rows["yoruba"].append({"llm": r["llm"], "flow": r["flow"], "ref": r["ref"],
                                   "zs": "ok" if r["verdict"] == "OK" else r["verdict"], "xl": "-"})
    out = {}
    print(f"{'language':16}{'serving checkpoint':26}{'mode':6}{'genders':>12}   (candidates that worked)")
    print("-" * 96)
    for lang in sorted(rows):
        p = pick(rows[lang])
        if not p:
            print(f"{lang:16}{'NONE WORKED':26}")
            continue
        llm, mode, cov = p
        working = sorted({r["llm"] for r in rows[lang]
                          if r["zs"] == "ok" or r["xl"] == "ok"}, key=epoch_key)
        out[lang] = {"llm": llm, "flow": rows[lang][0]["flow"], "mode":
                     "zero-shot" if mode == "zs" else "cross-lingual", "genders": cov}
        print(f"{lang:16}{llm.replace('_whole.pt',''):26}"
              f"{'zs' if mode=='zs' else 'xl':6}{cov:>12}   "
              f"{', '.join(w.replace('_whole.pt','') for w in working)}")
    json.dump(out, open("/leonardo_scratch/large/userexternal/atsado00/serving_checkpoints.json", "w"), indent=2)
    print(f"\nwritten serving_checkpoints.json ({len(out)} languages)")


if __name__ == "__main__":
    main()
