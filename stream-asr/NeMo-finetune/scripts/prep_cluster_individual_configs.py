import json
import yaml
from pathlib import Path
from omegaconf import OmegaConf

MANIFEST_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/manifests_clean")
CONFIG_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/configs")
BASE = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml"
CHECKPOINT = "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/nemotron-3.5-asr-streaming-0.6b.nemo"
TOKENIZER_DIR = "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/tokenizer/tokenizer_spe_unigram_v32000"

prompt_dict = yaml.safe_load(open(CONFIG_DIR / "prompt_dictionary.yaml"))["prompt_dictionary"]
summary = json.load(open(MANIFEST_DIR / "manifest_build_summary.json"))
agg_hours = {}
for bucket, v in summary.items():
    tl = v["target_lang"]
    agg_hours[tl] = agg_hours.get(tl, 0.0) + v["hours_written"]

# Well-evidenced 5-way semantic clustering, recovered from the omnivoice all_lab_tts_pipeline
# data_configs/clusters/*.json files, extended to cover our full 41-code set (that pipeline's
# exact later 7-way refinement's config files were lost in a July data incident and could not
# be fully recovered).
CLUSTERS = {
    "afroasiatic": ["ar-AR", "am-ET", "ti-ER", "ha-NG", "or-KE", "so-SO", "ber-MA"],
    "east_africa_bantu": ["ki-KE", "rw-RW", "lg-UG", "sw-KE"],
    "southern_central_bantu": ["ny-MW", "ln-CD", "nd-ZW", "nso-ZA", "st-ZA", "sn-ZW", "ss-SZ",
                                "ts-ZA", "tn-BW", "umb-AO", "ve-ZA", "xh-ZA", "zu-ZA", "bem-ZM"],
    "west_africa_niger_congo": ["bm-ML", "ee-GH", "fon-BJ", "ff-SN", "ig-NG", "tw-GH", "wo-SN", "yo-NG"],
    "other_languages": ["af-ZA", "kr-NG", "mg-MG", "kri-SL", "en-GH", "en-NG", "en-ZA", "en-UG"],
}


def input_cfg_for(codes, split):
    entries = []
    for tl in codes:
        p = MANIFEST_DIR / f"{tl}_{split}.jsonl"
        if p.exists() and p.stat().st_size > 0:
            entries.append({"type": "nemo", "manifest_filepath": str(p), "weight": 1.0,
                             "tags": {"target_lang": tl, "prompt_mode": "unified"}})
    return entries


def build_config(name, codes, out_root, max_steps, num_nodes, devices):
    cfg = OmegaConf.load(BASE)
    OmegaConf.set_struct(cfg, False)

    cfg.model.model_defaults.prompt_dictionary = prompt_dict
    cfg.model.model_defaults.num_prompts = 128

    cfg.init_from_nemo_model = CHECKPOINT
    cfg.init_from_nemo_model_exclude = ["decoder", "joint"]

    cfg.model.tokenizer.dir = TOKENIZER_DIR
    cfg.model.tokenizer.type = "bpe"
    cfg.model.tokenizer.update_tokenizer = True
    # See prep_final_config.py: NeMo's finetune script does hasattr(cfg.model.char_labels, ...)
    # without checking the key exists, which raises under struct mode for our BPE-only config.
    cfg.model.char_labels = None

    train_entries = input_cfg_for(codes, "train")
    dev_entries = input_cfg_for(codes, "dev")
    cfg.model.train_ds.manifest_filepath = None
    cfg.model.train_ds.input_cfg = train_entries
    cfg.model.train_ds.is_tarred = False
    cfg.model.train_ds.tarred_audio_filepaths = None
    # Some source datasets have stereo audio; channel_selector=0 is a no-op for mono
    # files and picks the left channel for stereo, fixing the (batch, channels, time)
    # vs expected (batch, time) shape mismatch at the dataloader level.
    cfg.model.train_ds.channel_selector = 0
    # The RNNT joint tensor scales with encoder_time x label_length x batch -- duration
    # bucketing alone doesn't bound the worst case tightly enough (confirmed empirically:
    # OOM'd at both 400s and 100s batch_duration). Capping max_duration bounds worst-case
    # encoder_time per utterance directly; combined with a smaller batch_duration this is
    # far more reliable than either alone.
    cfg.model.train_ds.batch_duration = 50
    cfg.model.train_ds.max_duration = 20.0
    # See prep_final_config.py: batch_duration/max_duration alone still let an outlier batch
    # OOM (confirmed on the combined run). quadratic_duration penalizes batches as utterances
    # exceed this length, shrinking batch size automatically -- matches NeMo's own test-fixture
    # convention for RNNT models.
    cfg.model.train_ds.quadratic_duration = 15.0
    # See prep_final_config.py: train_ds has no batch_size (count) cap by default -- a
    # batch of many short utterances plus one moderately-long one can stay under the
    # duration budget while still having a large count. Hard-cap as a backstop (cluster/
    # individual runs have been stable without it, but combined hit this concretely).
    cfg.model.train_ds.batch_size = 16
    cfg.model.validation_ds.manifest_filepath = None
    cfg.model.validation_ds.input_cfg = dev_entries if dev_entries else train_entries
    cfg.model.validation_ds.channel_selector = 0
    cfg.model.validation_ds.quadratic_duration = 15.0

    cfg.model.optim.name = "adamw"
    cfg.model.optim.lr = 1e-4
    cfg.model.optim.betas = [0.9, 0.98]
    cfg.model.optim.weight_decay = 1e-3
    cfg.model.optim.sched = OmegaConf.create({
        "name": "WarmupHoldPolicy", "warmup_ratio": 0.02, "hold_ratio": 0.97, "min_lr": 1e-6,
    })

    cfg.trainer.devices = devices
    cfg.trainer.num_nodes = num_nodes
    cfg.trainer.max_steps = max_steps
    cfg.trainer.max_epochs = -1
    cfg.trainer.val_check_interval = min(1000, max(100, max_steps // 10))
    cfg.trainer.limit_train_batches = None
    # Confirmed empirically via smoke test: unbounded validation iterates the full dev
    # set every val_check_interval steps and can run 20+ min without finishing one pass.
    cfg.trainer.limit_val_batches = 100
    cfg.trainer.enable_checkpointing = False
    cfg.trainer.logger = False

    cfg.exp_manager.exp_dir = str(out_root)
    cfg.exp_manager.name = name
    cfg.exp_manager.create_checkpoint_callback = True
    cfg.exp_manager.checkpoint_callback_params = OmegaConf.create({
        "save_top_k": -1, "every_n_train_steps": max(1000, max_steps // 5), "every_n_epochs": None,
        "monitor": "val_wer", "mode": "min", "save_best_model": False,
    })
    cfg.exp_manager.create_wandb_logger = True
    cfg.exp_manager.wandb_logger_kwargs = OmegaConf.create({
        "name": name, "project": "nemotron-asr-finetune", "entity": "john-eze",
    })
    cfg.exp_manager.resume_if_exists = True
    cfg.exp_manager.resume_ignore_no_checkpoint = True

    out_path = CONFIG_DIR / f"{name}_config.yaml"
    with open(out_path, "w") as f:
        OmegaConf.save(cfg, f)
    return out_path, len(train_entries), sum(agg_hours.get(c, 0.0) for c in codes)


def main():
    manifest_list = []

    # --- clusters: 4 nodes / 16 GPUs, step budget scaled loosely to cluster data volume ---
    for name, codes in CLUSTERS.items():
        hours = sum(agg_hours.get(c, 0.0) for c in codes)
        max_steps = int(min(60000, max(15000, hours * 15)))
        path, n_train, hours = build_config(f"cluster_{name}", codes, 
            f"/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/clusters/{name}",
            max_steps, num_nodes=1, devices=4)
        manifest_list.append({"kind": "cluster", "name": name, "config": str(path),
                               "codes": codes, "hours": round(hours, 1), "max_steps": max_steps})
        print(f"[cluster/{name}] {len(codes)} langs, {hours:.1f}h, max_steps={max_steps} -> {path}")

    # --- individual: 1 node / 4 GPUs, step budget scaled to that language's data volume ---
    for tl, hours in sorted(agg_hours.items(), key=lambda x: -x[1]):
        max_steps = int(min(40000, max(2000, hours * 15)))
        path, n_train, hours = build_config(f"individual_{tl}", [tl],
            f"/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/individual/{tl}",
            max_steps, num_nodes=1, devices=4)
        manifest_list.append({"kind": "individual", "name": tl, "config": str(path),
                               "codes": [tl], "hours": round(hours, 1), "max_steps": max_steps})
        print(f"[individual/{tl}] {hours:.1f}h, max_steps={max_steps} -> {path}")

    with open(CONFIG_DIR / "cluster_individual_manifest.json", "w") as f:
        json.dump(manifest_list, f, indent=2)
    print(f"\nWrote {len(manifest_list)} configs total")


if __name__ == "__main__":
    main()
