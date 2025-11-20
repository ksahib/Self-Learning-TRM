from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import os
import math
import yaml
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
from omegaconf import DictConfig
from pydantic import BaseModel

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class
from models.recursive_reasoning.metaTRM import MetaTRM
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.recursive_reasoning.helper import (
    compute_rewards_from_augmentations,
    load_base_trm_checkpoint,
    infer_vocab_size_from_checkpoint,
)


class MetaTrainConfig(BaseModel):
    # Meta model config
    meta_arch: Dict
    
    # Base model config
    base_checkpoint_path: str
    base_arch: Dict
    
    # Data
    data_paths: List[str]
    data_paths_val: List[str] = []
    
    # Hyperparams
    global_batch_size: int
    meta_epochs: int
    
    meta_lr: float
    meta_weight_decay: float = 0.01
    meta_beta1: float = 0.9
    meta_beta2: float = 0.95
    
    # Base fine-tuning
    num_finetune_steps: int = 10
    base_finetune_lr: float = 1e-4
    
    # Reward config
    reward_scale: float = 1.0
    use_binary_reward: bool = True
    
    # REINFORCE config
    baseline_momentum: float = 0.9
    entropy_coefficient: float = 0.01
    
    # Number of augmentation patterns to sample per batch
    num_patterns_per_batch: int = 3
    
    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    
    # Extras
    seed: int = 0
    checkpoint_interval: int = 100
    eval_interval: int = 10
    device: str = "cuda"
    loss_type: str = "softmax_cross_entropy"


@dataclass
class MetaTrainState:
    meta_model: nn.Module
    meta_optimizer: torch.optim.Optimizer
    baseline_reward: float
    step: int
    total_steps: int


def create_dataloader(config: MetaTrainConfig, split: str, **kwargs):
    dataset_paths = (
        config.data_paths_val if split == "val" and len(config.data_paths_val) > 0 else config.data_paths
    )
    dataset_cfg = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=dataset_paths,
        global_batch_size=kwargs.get("global_batch_size", config.global_batch_size),
        test_set_mode=(split != "train"),
        epochs_per_iter=kwargs.get("epochs_per_iter", 1),
        rank=kwargs.get("rank", 0),
        num_replicas=kwargs.get("num_replicas", 1),
    )
    dataset = PuzzleDataset(dataset_cfg, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_meta_model(config: MetaTrainConfig, train_metadata: PuzzleDatasetMetadata):
    """Create and initialize meta model."""
    model_cfg = dict(
        **config.meta_arch,
        batch_size=config.global_batch_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
    )
    
    model_cls = load_model_class(config.meta_arch["name"])
    
    with torch.device(config.device):
        model: nn.Module = model_cls(model_cfg)
        print(f"Meta model: {model}")
        
        # Load checkpoint if specified
        if config.load_checkpoint is not None:
            print(f"Loading meta model checkpoint from {config.load_checkpoint}")
            state_dict = torch.load(config.load_checkpoint, map_location=config.device)
            model.load_state_dict(state_dict, strict=False)
    
    return model


def load_base_model(config: MetaTrainConfig, train_metadata: PuzzleDatasetMetadata):
    """Load pretrained base TRM model."""
    base_arch_cfg = dict(config.base_arch)
    for reserved_key in ("batch_size", "vocab_size", "seq_len", "num_puzzle_identifiers", "causal"):
        base_arch_cfg.pop(reserved_key, None)
    
    model_cfg = dict(
        **base_arch_cfg,
        batch_size=config.global_batch_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,
    )
    
    # Create base model
    base_model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
    
    # Load checkpoint
    base_model = load_base_trm_checkpoint(
        checkpoint_path=config.base_checkpoint_path,
        model=base_model,
        map_location=config.device,
        strict=False,
    )
    
    # Explicitly move model to device to ensure all buffers are moved
    device_obj = torch.device(config.device)
    base_model = base_model.to(device_obj)
    
    # Freeze base weights, only LoRA will be trainable
    for name, param in base_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    print(f"Base model loaded. Trainable LoRA params: {sum(p.numel() for p in base_model.parameters() if p.requires_grad)}")
    
    return base_model


def init_meta_train_state(config: MetaTrainConfig, train_metadata: PuzzleDatasetMetadata):
    """Initialize meta training state."""
    # Estimated total training steps
    total_steps = int(config.meta_epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    
    # Create models
    meta_model = create_meta_model(config, train_metadata)
    meta_model = meta_model.to(config.device)
    
    # Optimizer
    meta_optimizer = torch.optim.Adam(
        meta_model.parameters(),
        lr=config.meta_lr,
        weight_decay=config.meta_weight_decay,
        betas=(config.meta_beta1, config.meta_beta2)
    )
    
    return MetaTrainState(
        meta_model=meta_model,
        meta_optimizer=meta_optimizer,
        baseline_reward=0.0,
        step=0,
        total_steps=total_steps,
    )


def sample_patterns_from_meta_model(
    meta_model: MetaTRM,
    batch: Dict[str, torch.Tensor],
    num_patterns: int,
    temperature: float = 1.0,
) -> Tuple[List[str], torch.Tensor]:
    """
    Sample multiple augmentation patterns from meta model.
    
    Args:
        meta_model: MetaTRM model
        batch: Input batch
        num_patterns: Number of patterns to sample
        temperature: Sampling temperature
    
    Returns:
        patterns: List of pattern strings
        log_probs: Tensor of log probabilities [num_patterns, slots]
    """
    # Run meta model forward pass
    meta_carry = meta_model.initial_carry(batch)
    meta_carry, meta_outputs = meta_model(
        meta_carry,
        batch,
        sample=True,
        temperature=temperature,
    )
    
    # Get patterns and log probs from batch
    all_patterns = meta_outputs["sampled_patterns"]  # List[str], one per batch element
    all_log_probs = meta_outputs["sampled_log_probs"]  # [batch, slots]
    
    # Take first num_patterns patterns (or cycle if needed)
    patterns = []
    log_probs_list = []
    for i in range(num_patterns):
        idx = i % len(all_patterns)
        patterns.append(all_patterns[idx])
        log_probs_list.append(all_log_probs[idx])
    
    log_probs = torch.stack(log_probs_list)  # [num_patterns, slots]
    
    return patterns, log_probs


def train_meta_batch(
    config: MetaTrainConfig,
    train_state: MetaTrainState,
    base_model: TinyRecursiveReasoningModel_ACTV1,
    train_batch: Dict[str, torch.Tensor],
    val_batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Train one meta batch using REINFORCE algorithm.
    
    Flow:
    1. Sample augmentation patterns from meta model
    2. Fine-tune base TRM on augmented data, evaluate on original
    3. Compute rewards
    4. REINFORCE policy gradient update
    """
    train_state.step += 1
    
    # Move batches to device
    train_batch = {k: v.to(config.device) for k, v in train_batch.items()}
    val_batch = {k: v.to(config.device) for k, v in val_batch.items()}
    
    # 1. Sample augmentation patterns from meta model
    patterns, log_probs = sample_patterns_from_meta_model(
        meta_model=train_state.meta_model,
        batch=train_batch,
        num_patterns=config.num_patterns_per_batch,
        temperature=1.0,
    )
    
    # log_probs: [num_patterns, slots]
    # Sum over slots to get total log prob per pattern
    pattern_log_probs = log_probs.sum(dim=-1)  # [num_patterns]
    
    # 2. Compute rewards for each pattern
    (
        rewards,
        baseline_loss,
        baseline_metrics,
        pattern_metrics,
    ) = compute_rewards_from_augmentations(
        base_model=base_model,
        original_batch=val_batch,  
        patterns=patterns,
        num_finetune_steps=config.num_finetune_steps,
        baseline_loss=None,  # Will compute automatically
        use_binary_reward=config.use_binary_reward,
        reward_scale=config.reward_scale,
        loss_type=config.loss_type,
    )
    
    # rewards: List[float], e.g., [0.0, 1.0, 0.0]
    rewards_tensor = torch.tensor(rewards, device=config.device, dtype=torch.float32)
    
    # 3. Update reward baseline (moving average)
    mean_reward = rewards_tensor.mean().item()
    train_state.baseline_reward = (
        train_state.baseline_reward * config.baseline_momentum +
        mean_reward * (1 - config.baseline_momentum)
    )
    
    # 4. Compute policy gradient (REINFORCE)
    advantages = rewards_tensor - train_state.baseline_reward  # [num_patterns]
    
    # Policy loss: -log_prob * advantage
    policy_loss = -(pattern_log_probs * advantages).mean()
    
    # 5. Entropy regularization (encourage exploration)
    # Compute entropy from log_probs
    probs = torch.exp(log_probs)  # [num_patterns, slots]
    entropy = -(probs * log_probs).sum(dim=-1).mean()  # Average entropy over patterns
    entropy_loss = -config.entropy_coefficient * entropy  # Negative because we want to maximize entropy
    
    total_loss = policy_loss + entropy_loss
    
    # 6. Backprop and update meta model
    train_state.meta_optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(train_state.meta_model.parameters(), max_norm=1.0)
    train_state.meta_optimizer.step()
    
    # 7. Metrics
    trm_metric_keys = [
        "loss",
        "accuracy",
        "exact_accuracy",
        "q_halt_accuracy",
        "lm_loss",
        "q_halt_loss",
        "q_continue_loss",
        "steps",
    ]
    
    trm_baseline_metrics = {
        f"trm_eval/baseline/{key}": baseline_metrics.get(key)
        for key in trm_metric_keys
    }
    
    trm_post_metrics: Dict[str, Optional[float]] = {}
    for key in trm_metric_keys:
        values = [m.get(key) for m in pattern_metrics if m.get(key) is not None]
        trm_post_metrics[f"trm_eval/post/{key}_mean"] = (
            float(sum(values) / len(values)) if len(values) > 0 else None
        )
    
    metrics = {
        "meta/policy_loss": policy_loss.item(),
        "meta/entropy": entropy.item(),
        "meta/entropy_loss": entropy_loss.item(),
        "meta/total_loss": total_loss.item(),
        "meta/mean_reward": mean_reward,
        "meta/baseline_reward": train_state.baseline_reward,
        "meta/baseline_loss": baseline_loss,
        "meta/rewards": rewards,  # List of rewards
        "meta/step": train_state.step,
        **trm_baseline_metrics,
        **trm_post_metrics,
    }
    
    return metrics


def save_checkpoint(config: MetaTrainConfig, train_state: MetaTrainState):
    """Save meta model checkpoint."""
    if config.checkpoint_path is None:
        return
    
    os.makedirs(config.checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(config.checkpoint_path, f"meta_step_{train_state.step}.pt")
    torch.save(train_state.meta_model.state_dict(), checkpoint_file)
    print(f"Saved checkpoint to {checkpoint_file}")


def load_synced_config(hydra_config: DictConfig) -> MetaTrainConfig:
    """Load and process config."""
    config = MetaTrainConfig(**hydra_config)  # type: ignore
    
    # Naming
    if config.project_name is None:
        config.project_name = "MetaTRM-REINFORCE"
    if config.run_name is None:
        config.run_name = f"meta-{coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)
    
    return config


@hydra.main(config_path="config", config_name="meta_train", version_base=None)
def launch(hydra_config: DictConfig):
    # Load config
    config = load_synced_config(hydra_config)
    
    # Seed
    torch.manual_seed(config.seed)
    
    # Device
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Dataset
    train_loader, train_metadata = create_dataloader(
        config, "train", rank=0, num_replicas=1, epochs_per_iter=1, global_batch_size=config.global_batch_size
    )
    
    val_loader = None
    val_metadata = None
    if len(config.data_paths_val) > 0:
        try:
            val_loader, val_metadata = create_dataloader(
                config, "val", rank=0, num_replicas=1, epochs_per_iter=1, global_batch_size=config.global_batch_size
            )
        except:
            print("Warning: Could not create validation loader, using train data for validation")
            val_loader, val_metadata = train_loader, train_metadata
    else:
        val_loader, val_metadata = train_loader, train_metadata
    
    # Initialize training state
    train_state = init_meta_train_state(config, train_metadata)
    
    # Load base model
    base_model = load_base_model(config, train_metadata)
    base_model = base_model.to(device)
    
    # Progress bar and logger
    progress_bar = tqdm.tqdm(total=train_state.total_steps)
    # Convert config to dict for wandb
    config_dict = config.dict() if hasattr(config, 'dict') else config.__dict__
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=config_dict,
        settings=wandb.Settings(_disable_stats=True)
    )
    wandb.log({"num_meta_params": sum(x.numel() for x in train_state.meta_model.parameters())}, step=0)
    
    # Training Loop
    print(f"Starting meta training for {config.meta_epochs} epochs...")
    
    for epoch in range(config.meta_epochs):
        print(f"\nEpoch {epoch + 1}/{config.meta_epochs}")
        
        train_state.meta_model.train()
        
        for batch_idx, (set_name, train_batch, global_batch_size) in enumerate(train_loader):
            # Get validation batch (cycle through validation set)
            try:
                val_iter = iter(val_loader)
                _, val_batch, _ = next(val_iter)
            except:
                val_batch = train_batch  # Fallback to train batch
            
            # Train meta batch
            metrics = train_meta_batch(
                config=config,
                train_state=train_state,
                base_model=base_model,
                train_batch=train_batch,
                val_batch=val_batch,
            )
            
            # Log metrics
            if train_state.step % 10 == 0:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(10)
            
            # Checkpoint
            if train_state.step % config.checkpoint_interval == 0:
                save_checkpoint(config, train_state)
            
            # Evaluation
            if train_state.step % config.eval_interval == 0 and train_state.step > 0:
                print(f"\nStep {train_state.step}: Mean reward = {metrics['meta/mean_reward']:.4f}, Baseline = {metrics['meta/baseline_reward']:.4f}")
        
        # Epoch checkpoint
        save_checkpoint(config, train_state)
    
    # Final checkpoint
    save_checkpoint(config, train_state)
    wandb.finish()
    print("Meta training completed!")


if __name__ == "__main__":
    launch()

