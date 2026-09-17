"""
One-time cleanup: deletes every per-stage-split CosyVoice3 repo (cosyvoice3-llm-*,
cosyvoice3-flow-*, cosyvoice3-hifigan-*) from the "CosyVoice3 TTS — African Language
Fine-tunes" collection and from HF entirely, keeping only the combined single-model repos
(cosyvoice3-individual-{lang}, one llm+flow+original-vocoder bundle per language). Confirmed
explicitly with the user, including for languages that have no combined replacement yet.
"""
import os

from huggingface_hub import HfApi

ORG = "all-lab"
COLLECTION_TITLE = "CosyVoice3 TTS — African Language Fine-tunes"


def main():
    token = os.environ.get("HF_TOKEN_ORGOWNER") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    print("authenticated as:", api.whoami().get("name"))

    collections = list(api.list_collections(owner=ORG))
    coll = next((c for c in collections if c.title == COLLECTION_TITLE), None)
    assert coll is not None, "collection not found"
    full = api.get_collection(coll.slug)

    to_delete = []
    to_keep = []
    for item in full.items:
        repo_id = item.item_id
        name = repo_id.split("/", 1)[1]
        if name.startswith("cosyvoice3-llm-") or name.startswith("cosyvoice3-flow-") or name.startswith("cosyvoice3-hifigan-"):
            to_delete.append(repo_id)
        else:
            to_keep.append(repo_id)

    print(f"\nkeeping {len(to_keep)}:")
    for r in to_keep:
        print(" ", r)
    print(f"\ndeleting {len(to_delete)}:")
    for r in to_delete:
        print(" ", r)

    print("\n=== deleting ===")
    failed = []
    for repo_id in to_delete:
        try:
            api.delete_repo(repo_id=repo_id, repo_type="model")
            print(f"[deleted] {repo_id}")
        except Exception as e:
            print(f"[FAILED] {repo_id}: {e}")
            failed.append(repo_id)

    print(f"\n=== done: {len(to_delete) - len(failed)} deleted, {len(failed)} failed ===")
    if failed:
        print("failed repos:", failed)


if __name__ == "__main__":
    main()
