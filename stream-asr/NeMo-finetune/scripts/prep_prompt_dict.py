"""Language-prompt slots for our 38 languages on nemotron-3.5-asr-streaming-0.6b.

The model conditions on a 128-slot one-hot vector. The source of truth for which slots are
taken is the prompt_dictionary inside the shipped .nemo, not the example yaml in the NeMo repo
(which differs), so this script reads it from the checkpoint and refuses to assign a slot that
is already used.

Three kinds of assignment:

  REUSE   -- codes NVIDIA already reserved in the shipped dictionary (sw-KE 48 ... or-KE 59,
             ar-AR 7). Twelve African codes have slots but were never trained; using them costs
             nothing and keeps our keys identical to NVIDIA's.
  ENGLISH -- one English. The corpus merges en-GH/NG/UG/ZA and the English moved out of other
             languages into code "en", which the shipped dictionary already maps to the trained
             English slot 0 (transcripts are tagged <en-US>, a tag the base tokenizer has).
  NEW     -- 24 languages get free slots. The numbers match the August run's, so its logs and
             manifests stay interpretable.

nd-ZW is kept as a key because our manifests use it, but every ndebele source is South African
Ndebele, so nr-ZA (its correct ISO 639-1 locale) is added as an alias to the same slot.

The result must also be written into the model's own cfg.model_defaults.prompt_dictionary
before saving: speech_to_text_finetune.py restores the config from the .nemo, and in August the
extended dictionary reached the dataloader but never the saved checkpoints.
"""
import tarfile
from pathlib import Path

import yaml

NEMO = "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/nemotron-3.5-asr-streaming-0.6b.nemo"
OUT = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/configs/prompt_dictionary_v2.yaml")
NUM_PROMPTS = 128

REUSE = ["sw-KE", "am-ET", "ha-NG", "zu-ZA", "yo-NG", "ig-NG", "af-ZA", "rw-RW",
         "so-SO", "ny-MW", "ln-CD", "or-KE", "ar-AR"]

ALIAS = {"nr-ZA": "nd-ZW"}

NEW = {
    "tw-GH": 63, "ee-GH": 72, "st-ZA": 73, "tn-BW": 74, "nd-ZW": 75, "ts-ZA": 76,
    "ff-SN": 77, "bm-ML": 78, "ki-KE": 79, "xh-ZA": 84, "ve-ZA": 85, "ti-ER": 86,
    "lg-UG": 87, "ss-SZ": 88, "bem-ZM": 89, "nso-ZA": 90, "mg-MG": 91, "sn-ZW": 92,
    "kr-NG": 93, "fon-BJ": 94, "kri-SL": 95, "ber-MA": 105, "wo-SN": 106, "umb-AO": 107,
}


def shipped_dictionary():
    with tarfile.open(NEMO) as tar:
        member = next(m for m in tar.getmembers() if m.name.lstrip("./") == "model_config.yaml")
        cfg = yaml.safe_load(tar.extractfile(member))
    return dict(cfg["model_defaults"]["prompt_dictionary"])


def main():
    shipped = shipped_dictionary()
    used = set(shipped.values())

    missing = [c for c in REUSE if c not in shipped]
    assert not missing, f"expected reserved codes absent from checkpoint: {missing}"

    clash = {c: i for c, i in NEW.items() if i in used}
    assert not clash, f"slots already used in the shipped dictionary: {clash}"
    assert len(set(NEW.values())) == len(NEW), "two new languages share a slot"
    assert all(0 <= i < NUM_PROMPTS for i in NEW.values())

    merged = {**shipped, **NEW}
    for alias, target in ALIAS.items():
        merged[alias] = merged[target]

    assert shipped.get("en") == 0, "expected the shipped dictionary to map 'en' to slot 0"
    ours = REUSE + list(NEW) + ["en"]
    assert len(set(ours)) == 38, len(set(ours))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("prompt_dictionary:\n")
        for k, v in merged.items():
            f.write(f"  {k}: {v}\n")

    free = sorted(set(range(NUM_PROMPTS)) - set(merged.values()))
    print(f"shipped: {len(shipped)} keys / {len(used)} slots")
    print(f"ours:    {len(REUSE)} reused, {len(NEW)} new slots, {len(ALIAS)} aliases")
    print(f"merged:  {len(merged)} keys / {len(set(merged.values()))} slots, {len(free)} still free: {free}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
