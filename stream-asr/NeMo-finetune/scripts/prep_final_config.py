import yaml
from omegaconf import OmegaConf

BASE = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml"
OUT = "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/configs/combined_train_config.yaml"
CONFIG_DIR = "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/configs"

cfg = OmegaConf.load(BASE)

prompt_dict = yaml.safe_load(open(f"{CONFIG_DIR}/prompt_dictionary.yaml"))["prompt_dictionary"]
train_blend = yaml.safe_load(open(f"{CONFIG_DIR}/train_blend.yaml"))["input_cfg"]
dev_blend = yaml.safe_load(open(f"{CONFIG_DIR}/dev_blend.yaml"))["input_cfg"]

# --- model_defaults / prompt dictionary ---
cfg.model.model_defaults.prompt_dictionary = prompt_dict
cfg.model.model_defaults.num_prompts = 128

# --- init from checkpoint, exclude decoder/joint (vocab size changed) ---
cfg.init_from_nemo_model = "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/nemotron-3.5-asr-streaming-0.6b.nemo"
OmegaConf.set_struct(cfg, False)
cfg.init_from_nemo_model_exclude = ["decoder", "joint"]

# --- tokenizer ---
cfg.model.tokenizer.dir = "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/tokenizer/tokenizer_spe_unigram_v32000"
cfg.model.tokenizer.type = "bpe"
cfg.model.tokenizer.update_tokenizer = True
# NeMo's speech_to_text_finetune.py does hasattr(cfg.model.char_labels, ...) without checking
# the key exists first; our BPE-tokenizer config has no char_labels section (correctly, since
# we're not char-based), which raises ConfigAttributeError under struct mode before hasattr can
# catch it. Adding the key as None makes hasattr(None, 'update_labels') cleanly return False.
cfg.model.char_labels = None

# --- data ---
# channel_selector=0: some source datasets (twi, hausa, ewe, others) have stereo audio;
# the model expects mono (batch, time). Selecting channel 0 is a no-op for already-mono
# files and picks the left channel for stereo ones -- fixes the shape mismatch at the
# dataloader level rather than needing to re-encode every audio file.
cfg.model.train_ds.manifest_filepath = None
cfg.model.train_ds.input_cfg = train_blend
cfg.model.train_ds.is_tarred = False
cfg.model.train_ds.tarred_audio_filepaths = None
cfg.model.train_ds.channel_selector = 0
# The RNNT joint tensor scales with encoder_time x label_length x batch, not just
# batch_duration -- duration-based bucketing alone doesn't bound the worst case tightly
# enough (confirmed empirically: OOM'd at both 400s and 100s, the second time at step 1
# trying to allocate 40GB on a 64GB GPU). Capping max_duration bounds the worst-case
# encoder_time per utterance directly; combined with a smaller batch_duration this bounds
# the joint tensor size far more reliably than either alone.
cfg.model.train_ds.batch_duration = 25
cfg.model.train_ds.max_duration = 10.0
# Confirmed empirically across FOUR OOMs at progressively tighter settings (400/15 -> 50/20/15 ->
# 35/14/12 -> 35/14/12+batch_size=16), the last two still OOM'd at ~31-38GiB for the RNNT joint's
# gradient buffer (torch.zeros_like(acts) in rnnt_pytorch.py). Back-of-envelope: joint tensor size
# is roughly batch x T_enc x U_label x vocab x 4 bytes; with vocab=32001, batch=16, max_duration=14
# (~175 encoder frames) and typical label lengths, the WORST CASE (all utterances near the cap
# simultaneously) still lands at ~30-38GiB even with a count cap -- the cap values themselves,
# not the mechanism, were too loose. Combined draws from all 41 temperature-reweighted sources,
# making it far more likely to hit this worst case than any single cluster/language (which have
# run stably for 1h+ at looser 50/20/15 settings) -- so combined needs a much lower ceiling on
# both duration and count to bound the tensor comfortably under the 63GB budget with headroom
# for model/optimizer/activation memory.
cfg.model.train_ds.quadratic_duration = 8.0
cfg.model.train_ds.batch_size = 8

cfg.model.validation_ds.manifest_filepath = None
cfg.model.validation_ds.input_cfg = dev_blend
cfg.model.validation_ds.channel_selector = 0
cfg.model.validation_ds.quadratic_duration = 12.0

# --- optimizer / WSD schedule: warmup 2%, then hold constant peak LR for the rest
# (no scheduled decay -- a clean cooldown phase gets added later, once, when we
# actually decide to stop, rather than baking a decay endpoint in now).
cfg.model.optim.name = "adamw"
cfg.model.optim.lr = 1e-4
cfg.model.optim.betas = [0.9, 0.98]
cfg.model.optim.weight_decay = 1e-3
cfg.model.optim.sched = OmegaConf.create({
    "name": "WarmupHoldPolicy",
    "warmup_ratio": 0.02,
    "hold_ratio": 0.97,
    "min_lr": 1e-6,
})

# --- trainer: multi-node ---
cfg.trainer.devices = 4
cfg.trainer.num_nodes = 8
cfg.trainer.max_steps = 100000
cfg.trainer.max_epochs = -1
cfg.trainer.val_check_interval = 2000
cfg.trainer.limit_train_batches = None
# Confirmed empirically via smoke test: with no limit, validation iterates the FULL dev
# set across all languages (9,646+ batches and still climbing after 22 min) every single
# val_check_interval steps -- would dominate wall-clock time and burn compute meant for
# training. Cap to a representative sample instead.
cfg.trainer.limit_val_batches = 100
cfg.trainer.enable_checkpointing = False  # exp_manager handles checkpointing, not the Trainer flag
cfg.trainer.logger = False  # exp_manager handles logging

# --- exp_manager: checkpointing every 10k steps, wandb ---
cfg.exp_manager.exp_dir = "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/combined"
cfg.exp_manager.name = "nemotron_combined_multilingual"
cfg.exp_manager.create_checkpoint_callback = True
cfg.exp_manager.checkpoint_callback_params = OmegaConf.create({
    "save_top_k": -1,
    "every_n_train_steps": 10000,
    "every_n_epochs": None,
    "monitor": "val_wer",
    "mode": "min",
    "save_best_model": False,
})
cfg.exp_manager.create_wandb_logger = True
cfg.exp_manager.wandb_logger_kwargs = OmegaConf.create({
    "name": "nemotron-combined-multilingual",
    "project": "nemotron-asr-finetune",
    "entity": "john-eze",
})
cfg.exp_manager.resume_if_exists = True
cfg.exp_manager.resume_ignore_no_checkpoint = True

with open(OUT, "w") as f:
    OmegaConf.save(cfg, f)

print(f"Wrote final config to {OUT}")
print(f"train sources: {len(train_blend)}, dev sources: {len(dev_blend)}")
