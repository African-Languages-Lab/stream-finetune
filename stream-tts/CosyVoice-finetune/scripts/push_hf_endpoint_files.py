"""
Adds Inference-Endpoint-readiness to each already-pushed proven model repo: vendors cosyvoice/
and third_party/Matcha-TTS/ (neither is pip-installable -- this project always uses them via
sys.path against a local checkout, so vendoring into the repo is what makes it self-contained
inside an endpoint container), plus handler.py (EndpointHandler, zero-shot cloning contract)
and requirements.txt. Only adds these small source/text files -- does not touch the existing
multi-GB weight files already uploaded by build_and_push_proven_hf_bundles.py.
"""
import argparse
import os
import shutil

from huggingface_hub import HfApi

COSYVOICE_SRC = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/cosyvoice"
MATCHA_SRC = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS"
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/hf_bundles_proven"

ORG = "all-lab"
LANGS = ["ha-NG", "tw-GH", "ig-NG", "ee-GH"]


def prep(lang):
    bundle_dir = f"{BUNDLE_ROOT}/{lang}"
    assert os.path.isdir(bundle_dir), f"{bundle_dir} missing -- run build_and_push_proven_hf_bundles.py first"

    dst_cosyvoice = f"{bundle_dir}/cosyvoice"
    if os.path.exists(dst_cosyvoice):
        shutil.rmtree(dst_cosyvoice)
    shutil.copytree(COSYVOICE_SRC, dst_cosyvoice, ignore=shutil.ignore_patterns("__pycache__"))

    dst_matcha = f"{bundle_dir}/third_party/Matcha-TTS"
    os.makedirs(f"{bundle_dir}/third_party", exist_ok=True)
    if os.path.exists(dst_matcha):
        shutil.rmtree(dst_matcha)
    # "data" is a broken symlink in the upstream checkout (points to the original author's
    # local machine, e.g. /home/smehta/...) -- unused by inference, just a leftover dataset
    # placeholder, so it's excluded rather than crashing copytree.
    shutil.copytree(MATCHA_SRC, dst_matcha, ignore=shutil.ignore_patterns("__pycache__", ".git", "data"))

    shutil.copy2(f"{SCRIPTS_DIR}/hf_endpoint_handler.py", f"{bundle_dir}/handler.py")
    shutil.copy2(f"{SCRIPTS_DIR}/hf_endpoint_requirements.txt", f"{bundle_dir}/requirements.txt")
    return bundle_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=LANGS)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN_ORGOWNER") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    print("authenticated as:", api.whoami().get("name"))

    for lang in args.langs:
        bundle_dir = prep(lang)
        repo_id = f"{ORG}/cosyvoice3-individual-{lang}"
        print(f"[upload endpoint files] {repo_id}")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=bundle_dir,
            repo_type="model",
            allow_patterns=["cosyvoice/**", "third_party/**", "handler.py", "requirements.txt"],
        )
        print(f"[done] {repo_id}")


if __name__ == "__main__":
    main()
