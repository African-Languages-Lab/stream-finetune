"""
Pushes each fully-trained CosyVoice3 regime as ONE self-contained model repo (llm.pt + flow.pt
+ hift.pt + shared config/tokenizer/campplus assets), matching the official release's own
directory layout -- loadable directly via CosyVoice3(model_dir), no manual assembly needed.
Replaces the earlier per-stage repo split (cosyvoice3-llm-*, -flow-*, -hifigan-*) for the 4
regimes that actually have all three stages genuinely trained.
"""
from pathlib import Path

from huggingface_hub import HfApi

ORG = "all-lab"
BUNDLE_ROOT = Path("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/hf_bundles")
LANGS = ["en-UG", "ber-MA", "umb-AO", "wo-SN"]

COLLECTION_TITLE = "CosyVoice3 TTS — African Language Fine-tunes"


def main():
    api = HfApi()
    collections = list(api.list_collections(owner=ORG))
    coll = next((c for c in collections if c.title == COLLECTION_TITLE), None)
    assert coll is not None, "expected collection to already exist from the earlier per-stage push"

    for lang in LANGS:
        repo_id = f"{ORG}/cosyvoice3-individual-{lang}"
        print(f"[upload] {repo_id} <- {BUNDLE_ROOT / lang}")
        api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
        api.upload_folder(repo_id=repo_id, folder_path=str(BUNDLE_ROOT / lang), repo_type="model")
        api.add_collection_item(coll.slug, item_id=repo_id, item_type="model", exists_ok=True)
        print(f"[done] {repo_id}")


if __name__ == "__main__":
    main()
