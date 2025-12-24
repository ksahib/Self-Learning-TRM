from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import os
import math
import time
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
from utils.vector_db import PuzzleVectorDB
from few_shot_dataset import FewShotDataset


class MetaTrainConfig(BaseModel):
    # Meta model config
    meta_arch: Dict
    
    # Base model config
    base_checkpoint_path: str
    base_arch: Dict
    
    # Data
    data_paths: List[str]
    data_paths_val: List[str] = []
    data_paths_test: List[str] = []
    
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
    # Compute-aware H/L cost trade-off for continuous rewards
    hl_cost_lambda: float = 0.01  # Penalty strength λ
    hl_cost_alpha: float = 1.0    # Weight for H_cycles
    hl_cost_beta: float = 1.0     # Weight for L_cycles
    
    # REINFORCE config
    baseline_momentum: float = 0.9
    entropy_coefficient: float = 0.01
    # GRPO/PPO-style config
    ppo_clip_epsilon: float = 0.2
    use_kl_penalty: bool = True
    kl_target: float = 0.01
    kl_beta_init: float = 1.0
    kl_adaptation_rate: float = 2.0
    # Reference model config (for two-stage training)
    use_reference_model: bool = False  # If False, use PPO-style (old_log_probs). If True, use reference model GRPO
    reference_checkpoint: Optional[str] = None  # Load reference model from checkpoint (for stage 2)
    switch_to_reference_at_step: Optional[int] = None  # Auto-switch to reference model at this step (None = manual)
    # Reference model update (only used if use_reference_model=True)
    use_ref_ema: bool = True
    ref_ema_decay: float = 0.995
    ref_update_interval: int = 500
    
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
    # Logging cadence
    # How often to log per-step training metrics (in meta steps). Set to 1 to log every step.
    log_interval: int = 1
    # In evaluate_meta, how many eval batches to aggregate per logged point.
    # This controls how dense the eval curves are in W&B (e.g., 10 → log every 10 eval batches).
    eval_log_step: int = 10
    device: str = "cuda"
    loss_type: str = "softmax_cross_entropy"
    
    # Few-shot training (SEAL-style)
    use_few_shot: bool = False
    num_similar_examples: int = 3
    vector_db_path: Optional[str] = None
    rebuild_vector_db: bool = False
    meta_trm_config: Optional[Dict] = None
    meta_trm_checkpoint: Optional[str] = None
    grid_height: int = 9
    grid_width: int = 9
    
    # Eval-only mode
    eval_only: bool = False
    eval_num_batches: int = 100  # Number of batches to evaluate over for stable metrics


@dataclass
class MetaTrainState:
    meta_model: nn.Module
    meta_optimizer: torch.optim.Optimizer
    meta_ref_model: Optional[nn.Module]  # Optional: only created if use_reference_model=True
    baseline_reward: float
    step: int
    total_steps: int
    kl_beta: float


def create_dataloader(
    config: MetaTrainConfig,
    split: str,
    vector_db: Optional[PuzzleVectorDB] = None,
    meta_model: Optional[MetaTRM] = None,
    base_model: Optional[TinyRecursiveReasoningModel_ACTV1] = None,
    **kwargs
):
    if split == "val" and len(config.data_paths_val) > 0:
        dataset_paths = config.data_paths_val
    elif split != "train" and len(config.data_paths_test) > 0:
        dataset_paths = config.data_paths_test
    else:
        dataset_paths = config.data_paths
    dataset_cfg = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=dataset_paths,
        global_batch_size=kwargs.get("global_batch_size", config.global_batch_size),
        test_set_mode=(split != "train"),
        epochs_per_iter=kwargs.get("epochs_per_iter", 1),
        rank=kwargs.get("rank", 0),
        num_replicas=kwargs.get("num_replicas", 1),
    )
    base_dataset = PuzzleDataset(dataset_cfg, split=split)
    
    # Wrap with FewShotDataset if enabled (for any split, not just train)
    # This allows few-shot to work during evaluation on test/val data
    if config.use_few_shot and vector_db is not None and meta_model is not None and base_model is not None:
        dataset = FewShotDataset(
            base_dataset=base_dataset,
            vector_db=vector_db,
            meta_model=meta_model,
            base_model=base_model,
            num_similar_examples=config.num_similar_examples,
            device=config.device,
            grid_height=config.grid_height,
            grid_width=config.grid_width,
        )
    else:
        dataset = base_dataset
    
    # Use num_workers=0 when using few-shot dataset to avoid CUDA initialization issues in worker processes
    # Few-shot dataset needs to run models (base_model, meta_model) which requires CUDA
    # CUDA cannot be initialized in forked worker processes
    use_workers = 0 if config.use_few_shot and vector_db is not None and meta_model is not None and base_model is not None else 1
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=use_workers,
        prefetch_factor=8 if use_workers > 0 else None,
        pin_memory=True if use_workers > 0 else False,
        persistent_workers=True if use_workers > 0 else False
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
    
    # The base TRM uses CastedSparseEmbedding, which is configured with a fixed
    # max batch size. During meta-training we may create augmented batches
    # larger than the logical global_batch_size (e.g., one example per active
    # augmentation symbol, or flattened few-shot examples). To avoid runtime
    # errors like:
    #   "CastedSparseEmbedding received batch size X larger than
    #    configured max batch size Y",
    # we configure the base model with an effective batch size large enough
    # to cover both augmentation expansion and few-shot flattening.
    max_aug_per_puzzle = 6  # PATTERN_LENGTH for augmentation patterns
    max_few_shot = config.num_similar_examples if config.use_few_shot else 1
    effective_batch_size = config.global_batch_size * max(max_aug_per_puzzle, max_few_shot)

    model_cfg = dict(
        **base_arch_cfg,
        batch_size=effective_batch_size,
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
    
    # Reference model (optional, only created if use_reference_model=True)
    meta_ref_model = None
    if config.use_reference_model:
        if config.reference_checkpoint is not None:
            # Load reference model from checkpoint (for stage 2 training)
            print(f"Loading reference model from checkpoint: {config.reference_checkpoint}")
            meta_ref_model = create_meta_model(config, train_metadata)
            meta_ref_model = meta_ref_model.to(config.device)
            state_dict = torch.load(config.reference_checkpoint, map_location=config.device)
            meta_ref_model.load_state_dict(state_dict, strict=False)
            print("Reference model loaded from checkpoint")
        else:
            # Create reference model as deep copy (for stage 1 or if no checkpoint provided)
            meta_ref_model = copy.deepcopy(meta_model).to(config.device)
            print("Reference model created as deep copy of current model")
        
        meta_ref_model.eval()
        for p in meta_ref_model.parameters():
            p.requires_grad = False
        
        # Initialize buffers with a dummy forward pass
        with torch.no_grad():
            dummy_batch = {
                "inputs": torch.zeros(1, train_metadata.seq_len, dtype=torch.long, device=config.device),
                "labels": torch.zeros(1, train_metadata.seq_len, dtype=torch.long, device=config.device),
                "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=config.device),
            }
            try:
                dummy_carry = meta_ref_model.initial_carry(dummy_batch)
                _, _ = meta_ref_model(dummy_carry, dummy_batch, sample=False, temperature=1.0, greedy=False)
                print("Reference model buffers initialized")
            except Exception as e:
                print(f"Warning: Could not initialize reference model buffers: {e}")
    else:
        print("Using PPO-style GRPO (no reference model)")
    
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
        meta_ref_model=meta_ref_model,
        baseline_reward=0.0,
        step=0,
        total_steps=total_steps,
        kl_beta=config.kl_beta_init,
    )


def sample_patterns_from_meta_model(
    meta_model: MetaTRM,
    batch: Dict[str, torch.Tensor],
    num_patterns: int,
    temperature: float = 1.0,
) -> Tuple[
    List[str],
    torch.Tensor,
    List[int],
    List[int],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Sample multiple augmentation patterns and H/L cycle choices from the meta model.
    
    Args:
        meta_model: MetaTRM model
        batch: Input batch
        num_patterns: Number of patterns to sample
        temperature: Sampling temperature
    
    Returns:
        patterns: List of pattern strings (length = num_patterns)
        slot_log_probs: Tensor of per-slot log probabilities [num_patterns, slots]
        h_values: List of chosen H_cycles (length = num_patterns)
        l_values: List of chosen L_cycles (length = num_patterns)
        total_log_probs: Tensor of total log probabilities per pattern
            [num_patterns], equal to
            slot_log_probs[i].sum() + h_log_prob[i] + l_log_prob[i]
        sampled_indices: Tensor of sampled augmentation indices [num_patterns, slots]
        h_indices: Tensor of sampled H indices [num_patterns]
        l_indices: Tensor of sampled L indices [num_patterns]
        aug_logits: Raw augmentation logits for the chosen patterns
            [num_patterns, slots, choices]
        h_logits: Raw H-cycle logits for the chosen patterns [num_patterns, choices]
        l_logits: Raw L-cycle logits for the chosen patterns [num_patterns, choices]
    """
    # Run meta model forward pass
    meta_carry = meta_model.initial_carry(batch)
    meta_carry, meta_outputs = meta_model(
        meta_carry,
        batch,
        sample=True,
        temperature=temperature,
    )
    
    all_sampled_indices = meta_outputs.get("sampled_indices")  # [batch, slots]
    all_h_indices = meta_outputs.get("sampled_H_indices")      # [batch]
    all_l_indices = meta_outputs.get("sampled_L_indices")      # [batch]
    all_aug_logits = meta_outputs.get("aug_logits")            # [batch, slots, choices]
    all_h_logits = meta_outputs.get("h_logits")                # [batch, h_choices]
    all_l_logits = meta_outputs.get("l_logits")                # [batch, l_choices]

    # Get patterns, H/L values, and log probs from batch
    all_patterns = meta_outputs["sampled_patterns"]  # List[str], one per batch element
    all_slot_log_probs = meta_outputs["sampled_log_probs"]  # [batch, slots]
    all_h_values = meta_outputs["sampled_H_values"]  # [batch]
    all_l_values = meta_outputs["sampled_L_values"]  # [batch]
    all_h_log_probs = meta_outputs["sampled_H_log_probs"]  # [batch]
    all_l_log_probs = meta_outputs["sampled_L_log_probs"]  # [batch]
    
    # Take first num_patterns patterns (or cycle if needed)
    patterns: List[str] = []
    slot_log_probs_list: List[torch.Tensor] = []
    h_values: List[int] = []
    l_values: List[int] = []
    total_log_probs_list: List[torch.Tensor] = []
    sampled_indices_list: List[torch.Tensor] = []
    h_indices_list: List[torch.Tensor] = []
    l_indices_list: List[torch.Tensor] = []
    aug_logits_list: List[torch.Tensor] = []
    h_logits_list: List[torch.Tensor] = []
    l_logits_list: List[torch.Tensor] = []
    num_available = max(len(all_patterns), 1)
    
    for i in range(num_patterns):
        idx = i % num_available
        patterns.append(all_patterns[idx])
        slot_log_probs = all_slot_log_probs[idx]
        h_val = int(all_h_values[idx].item())
        l_val = int(all_l_values[idx].item())
        h_lp = all_h_log_probs[idx]
        l_lp = all_l_log_probs[idx]
        
        slot_log_probs_list.append(slot_log_probs)
        h_values.append(h_val)
        l_values.append(l_val)
        total_log_probs_list.append(slot_log_probs.sum() + h_lp + l_lp)
        if all_sampled_indices is not None:
            sampled_indices_list.append(all_sampled_indices[idx])
        if all_h_indices is not None:
            h_indices_list.append(all_h_indices[idx])
        if all_l_indices is not None:
            l_indices_list.append(all_l_indices[idx])
        if all_aug_logits is not None:
            aug_logits_list.append(all_aug_logits[idx])
        if all_h_logits is not None:
            h_logits_list.append(all_h_logits[idx])
        if all_l_logits is not None:
            l_logits_list.append(all_l_logits[idx])
    
    slot_log_probs_tensor = torch.stack(slot_log_probs_list)  # [num_patterns, slots]
    total_log_probs_tensor = torch.stack(total_log_probs_list)  # [num_patterns]
    sampled_indices_tensor = (
        torch.stack(sampled_indices_list) if len(sampled_indices_list) > 0 else None
    )
    h_indices_tensor = torch.stack(h_indices_list) if len(h_indices_list) > 0 else None
    l_indices_tensor = torch.stack(l_indices_list) if len(l_indices_list) > 0 else None
    aug_logits_tensor = torch.stack(aug_logits_list) if len(aug_logits_list) > 0 else None
    h_logits_tensor = torch.stack(h_logits_list) if len(h_logits_list) > 0 else None
    l_logits_tensor = torch.stack(l_logits_list) if len(l_logits_list) > 0 else None
    
    return (
        patterns,
        slot_log_probs_tensor,
        h_values,
        l_values,
        total_log_probs_tensor,
        sampled_indices_tensor,
        h_indices_tensor,
        l_indices_tensor,
        aug_logits_tensor,
        h_logits_tensor,
        l_logits_tensor,
    )


def train_meta_batch(
    config: MetaTrainConfig,
    train_state: MetaTrainState,
    base_model: TinyRecursiveReasoningModel_ACTV1,
    train_batch: Dict[str, torch.Tensor],
    val_batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Train one meta batch using GRPO/PPO-style policy gradient with a lagging
    reference policy.
    
    Flow:
    1. Sample augmentation patterns from meta model
    2. Fine-tune base TRM on augmented data, evaluate on original
    3. Compute rewards
    4. GRPO/PPO policy gradient update with ratio clipping + KL-to-reference
    """
    train_state.step += 1
    
    # Move batches to device
    train_batch = {k: v.to(config.device) for k, v in train_batch.items()}
    val_batch = {k: v.to(config.device) for k, v in val_batch.items()}
    
    # Handle few-shot batches: extract original puzzles for MetaTRM pattern sampling
    if config.use_few_shot and "similar_inputs" in train_batch:
        # Few-shot mode: use original puzzles for MetaTRM pattern sampling
        # The similar examples are already augmented and will be used during reward computation
        meta_batch = {
            "inputs": train_batch["inputs"],
            "labels": train_batch["labels"],
            "puzzle_identifiers": train_batch["puzzle_identifiers"],
        }
    else:
        meta_batch = train_batch
    
    # 1. Sample augmentation patterns and H/L cycles from meta model
    (
        patterns,
        slot_log_probs,
        h_values,
        l_values,
        pattern_log_probs,
        sampled_indices,
        h_indices,
        l_indices,
        aug_logits,
        h_logits,
        l_logits,
    ) = sample_patterns_from_meta_model(
        meta_model=train_state.meta_model,
        batch=meta_batch,
        num_patterns=config.num_patterns_per_batch,
        temperature=1.0,
    )
    
    # 2. Compute rewards for each pattern
    # In few-shot mode, pass similar examples for fine-tuning
    few_shot_train_batch = None
    if config.use_few_shot and "similar_inputs" in train_batch:
        # Extract similar examples for fine-tuning
        similar_inputs = train_batch["similar_inputs"]  # [batch_size, num_similar, seq_len]
        similar_labels = train_batch["similar_labels"]  # [batch_size, num_similar, seq_len]
        batch_size = similar_inputs.shape[0]
        num_similar = similar_inputs.shape[1]
        
        # Only create few-shot batch if we have actual similar examples
        if num_similar > 0 and similar_inputs.numel() > 0:
            # Flatten similar examples for training
            similar_inputs_flat = similar_inputs.reshape(-1, similar_inputs.shape[-1])  # Changed from view to reshape for PyTorch 2.8.0 compatibility
            similar_labels_flat = similar_labels.reshape(-1, similar_labels.shape[-1])  # Changed from view to reshape for PyTorch 2.8.0 compatibility
            
            # Use actual puzzle identifiers from similar examples if available, otherwise fallback to original puzzle IDs
            if "similar_puzzle_identifiers" in train_batch:
                similar_puzzle_ids = train_batch["similar_puzzle_identifiers"]  # [batch_size, num_similar]
                similar_puzzle_ids_flat = similar_puzzle_ids.reshape(-1)  # [batch_size * num_similar]
            else:
                # Fallback: use original puzzle identifiers (for backward compatibility)
                similar_puzzle_ids_flat = train_batch["puzzle_identifiers"].repeat_interleave(num_similar, dim=0)
            
            # Verify we have non-empty tensors with valid puzzle identifiers
            if similar_inputs_flat.shape[0] > 0 and similar_puzzle_ids_flat.shape[0] > 0:
                # Additional check: ensure puzzle identifiers are valid (within bounds)
                # Get num_puzzle_identifiers from base model config
                max_puzzle_id = base_model.config.num_puzzle_identifiers - 1
                if similar_puzzle_ids_flat.min() >= 0 and similar_puzzle_ids_flat.max() <= max_puzzle_id:
                    few_shot_train_batch = {
                        "inputs": similar_inputs_flat,
                        "labels": similar_labels_flat,
                        "puzzle_identifiers": similar_puzzle_ids_flat,
                    }
                else:
                    # Invalid puzzle identifiers, skip few-shot batch
                    print(f"Warning: Invalid puzzle identifiers detected (min={similar_puzzle_ids_flat.min()}, max={similar_puzzle_ids_flat.max()}, allowed=[0, {max_puzzle_id}]). Skipping few-shot batch.")
        # If empty or invalid, few_shot_train_batch remains None and we'll use standard augmentation
    
    (
        rewards,
        baseline_loss,
        baseline_metrics,
        pattern_metrics,
    ) = compute_rewards_from_augmentations(
        base_model=base_model,
        original_batch=meta_batch,  # Use same input for baseline and evaluation (matches pattern sampling)
        patterns=patterns,
        num_finetune_steps=config.num_finetune_steps,
        baseline_loss=None,  # Will compute automatically
        use_binary_reward=config.use_binary_reward,
        reward_scale=config.reward_scale,
        loss_type=config.loss_type,
        grid_height=config.grid_height,
        grid_width=config.grid_width,
        few_shot_train_batch=few_shot_train_batch,
        meta_model=train_state.meta_model if config.use_few_shot else None,
        h_cycles=h_values,
        l_cycles=l_values,
        hl_cost_lambda=config.hl_cost_lambda,
        hl_cost_alpha=config.hl_cost_alpha,
        hl_cost_beta=config.hl_cost_beta,
    )
    
    # rewards: List[float], e.g., [0.0, 1.0, 0.0]
    rewards_tensor = torch.tensor(rewards, device=config.device, dtype=torch.float32)
    
    # 3. Update reward baseline (moving average)
    mean_reward = rewards_tensor.mean().item()
    train_state.baseline_reward = (
        train_state.baseline_reward * config.baseline_momentum +
        mean_reward * (1 - config.baseline_momentum)
    )
    
    # Print pattern information during training (same format as eval)
    if train_state.step % max(config.log_interval, 1) == 0:  # Only print when logging to reduce noise
        baseline_acc = baseline_metrics.get("exact_accuracy", 0.0)
        baseline_loss_val = baseline_loss
        
        print(f"  Step {train_state.step} - Baseline: acc={baseline_acc:.4f}, loss={baseline_loss_val:.4f}")
        
        # Analyze each pattern's performance
        improved_patterns = []
        worsened_patterns = []
        for i, (pattern, pattern_metric) in enumerate(zip(patterns, pattern_metrics)):
            pattern_acc = pattern_metric.get("exact_accuracy", 0.0)
            pattern_loss = pattern_metric.get("loss", float('inf'))
            reward_val = rewards[i] if i < len(rewards) else 0.0
            improvement = pattern_acc - baseline_acc
            loss_improvement = baseline_loss_val - pattern_loss  # Positive means loss decreased (better)
            
            # Get H_cycle and L_cycle for this pattern
            h_cycle = h_values[i] if i < len(h_values) else None
            l_cycle = l_values[i] if i < len(l_values) else None
            
            status = "✓" if improvement > 0 else "✗"
            cycle_info = f"H={h_cycle},L={l_cycle}" if h_cycle is not None and l_cycle is not None else ""
            print(f"    Pattern {i+1} [{pattern}] {cycle_info}: acc={pattern_acc:.4f} ({improvement:+.4f}), "
                  f"loss={pattern_loss:.4f} ({loss_improvement:+.4f}), reward={reward_val:.2f} {status}")
            
            if improvement > 0:
                improved_patterns.append((pattern, improvement, pattern_acc))
            elif improvement < 0:
                worsened_patterns.append((pattern, improvement, pattern_acc))
        
        # Summary
        if improved_patterns:
            print(f"    ✓ Improved patterns ({len(improved_patterns)}): {[p[0] for p in improved_patterns]}")
        if worsened_patterns:
            print(f"    ✗ Worsened patterns ({len(worsened_patterns)}): {[p[0] for p in worsened_patterns]}")
        
        # Compute mean post-augmentation accuracy
        post_accs = [pm.get("exact_accuracy", 0.0) for pm in pattern_metrics]
        mean_post_acc = sum(post_accs) / len(post_accs) if len(post_accs) > 0 else 0.0
        overall_improvement = mean_post_acc - baseline_acc
        print(f"    Overall: mean_post_acc={mean_post_acc:.4f}, improvement={overall_improvement:+.4f}")
    
    # 4. Compute policy gradient (GRPO/PPO)
    def _gather_log_probs(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Utility to gather log probs for provided indices."""
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return log_probs.gather(dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)

    # Determine which mode to use: PPO-style (old_log_probs) or reference-model GRPO
    should_use_reference = (
        config.use_reference_model 
        and train_state.meta_ref_model is not None
        and (config.switch_to_reference_at_step is None or train_state.step >= config.switch_to_reference_at_step)
    )
    
    # Store old_log_probs for PPO-style (always store, use if not using reference model)
    old_log_probs = pattern_log_probs.detach()
    
    # Compute reference log-probs
    if should_use_reference:
        # Use reference model GRPO
        use_current_as_ref = False
        with torch.no_grad():
            # Ensure reference model is in eval mode
            train_state.meta_ref_model.eval()
        
        try:
            ref_carry = train_state.meta_ref_model.initial_carry(meta_batch)
            ref_carry, ref_outputs = train_state.meta_ref_model(
                ref_carry,
                meta_batch,
                sample=False,
                temperature=1.0,
                greedy=False,
            )
            ref_aug_logits = ref_outputs.get("aug_logits")
            ref_h_logits = ref_outputs.get("h_logits")
            ref_l_logits = ref_outputs.get("l_logits")
            
            # Check if outputs are valid
            if ref_aug_logits is None or ref_h_logits is None or ref_l_logits is None:
                raise ValueError("Reference model returned None logits")
            if torch.isnan(ref_aug_logits).any() or torch.isnan(ref_h_logits).any() or torch.isnan(ref_l_logits).any():
                raise ValueError("Reference model returned NaN logits")
        except Exception as e:
            if train_state.step == 1:
                print(f"Warning: Reference model failed on first step: {e}. Using current model as reference (ratio=1.0).")
                use_current_as_ref = True
                # Use current model's logits as reference
                ref_aug_logits = aug_logits.detach() if aug_logits is not None else None
                ref_h_logits = h_logits.detach() if h_logits is not None else None
                ref_l_logits = l_logits.detach() if l_logits is not None else None
            else:
                print(f"Warning: Reference model forward pass failed: {e}. Using zeros for ref log_probs.")
                ref_aug_logits = None
                ref_h_logits = None
                ref_l_logits = None
        
        # Check for NaN/inf in reference logits
        if ref_aug_logits is not None:
            if torch.isnan(ref_aug_logits).any() or torch.isinf(ref_aug_logits).any():
                print(f"Warning: NaN/inf in ref_aug_logits. Replacing with zeros.")
                ref_aug_logits = torch.zeros_like(ref_aug_logits)
        if ref_h_logits is not None:
            if torch.isnan(ref_h_logits).any() or torch.isinf(ref_h_logits).any():
                print(f"Warning: NaN/inf in ref_h_logits. Replacing with zeros.")
                ref_h_logits = torch.zeros_like(ref_h_logits)
        if ref_l_logits is not None:
            if torch.isnan(ref_l_logits).any() or torch.isinf(ref_l_logits).any():
                print(f"Warning: NaN/inf in ref_l_logits. Replacing with zeros.")
                ref_l_logits = torch.zeros_like(ref_l_logits)

        if use_current_as_ref:
            # On first step if ref model fails, use current model's log_probs as reference (ratio = 1.0)
            ref_pattern_log_probs = pattern_log_probs.detach()
            ref_aug_logits_sel = aug_logits.detach() if aug_logits is not None else None
            ref_h_logits_sel = h_logits.detach() if h_logits is not None else None
            ref_l_logits_sel = l_logits.detach() if l_logits is not None else None
        elif sampled_indices is None or sampled_indices.numel() == 0:
            ref_pattern_log_probs = torch.zeros_like(pattern_log_probs)
            ref_aug_logits_sel = None
            ref_h_logits_sel = None
            ref_l_logits_sel = None
        elif ref_aug_logits is None or ref_h_logits is None or ref_l_logits is None:
            # If reference model failed, use zeros
            num_patterns = pattern_log_probs.shape[0]
            ref_pattern_log_probs = torch.zeros(num_patterns, device=config.device)
            ref_aug_logits_sel = None
            ref_h_logits_sel = None
            ref_l_logits_sel = None
        else:
            # Subselect ref logits for the same pattern indices we kept above
            # so shapes match [num_patterns, ...]
            num_patterns = sampled_indices.shape[0]
            # Ensure we don't index beyond available logits
            batch_size = ref_aug_logits.shape[0]
            num_to_use = min(num_patterns, batch_size)
            ref_aug_logits_sel = ref_aug_logits[:num_to_use]
            ref_h_logits_sel = ref_h_logits[:num_to_use]
            ref_l_logits_sel = ref_l_logits[:num_to_use]
            
            # If we need more patterns than available, pad with zeros
            if num_to_use < num_patterns:
                pad_size = num_patterns - num_to_use
                ref_aug_logits_sel = torch.cat([
                    ref_aug_logits_sel,
                    torch.zeros(pad_size, *ref_aug_logits_sel.shape[1:], device=ref_aug_logits_sel.device, dtype=ref_aug_logits_sel.dtype)
                ], dim=0)
                ref_h_logits_sel = torch.cat([
                    ref_h_logits_sel,
                    torch.zeros(pad_size, *ref_h_logits_sel.shape[1:], device=ref_h_logits_sel.device, dtype=ref_h_logits_sel.dtype)
                ], dim=0)
                ref_l_logits_sel = torch.cat([
                    ref_l_logits_sel,
                    torch.zeros(pad_size, *ref_l_logits_sel.shape[1:], device=ref_l_logits_sel.device, dtype=ref_l_logits_sel.dtype)
                ], dim=0)

            # Compute reference log_probs with safety checks
            try:
                if ref_aug_logits_sel is not None and sampled_indices is not None:
                    # Clamp indices to valid range
                    max_idx = ref_aug_logits_sel.shape[-1] - 1
                    sampled_indices_safe = torch.clamp(sampled_indices, min=0, max=max_idx)
                    ref_slot_log_probs = _gather_log_probs(ref_aug_logits_sel, sampled_indices_safe)
                    # Check for NaN/inf
                    if torch.isnan(ref_slot_log_probs).any() or torch.isinf(ref_slot_log_probs).any():
                        print(f"Warning: NaN/inf in ref_slot_log_probs. Replacing with zeros.")
                        ref_slot_log_probs = torch.zeros_like(ref_slot_log_probs)
                else:
                    ref_slot_log_probs = torch.zeros(num_patterns, device=config.device)
                
                if ref_h_logits_sel is not None and h_indices is not None:
                    max_idx = ref_h_logits_sel.shape[-1] - 1
                    h_indices_safe = torch.clamp(h_indices, min=0, max=max_idx)
                    ref_h_log_probs = _gather_log_probs(ref_h_logits_sel, h_indices_safe)
                    if torch.isnan(ref_h_log_probs).any() or torch.isinf(ref_h_log_probs).any():
                        print(f"Warning: NaN/inf in ref_h_log_probs. Replacing with zeros.")
                        ref_h_log_probs = torch.zeros_like(ref_h_log_probs)
                else:
                    ref_h_log_probs = torch.zeros(num_patterns, device=config.device)
                
                if ref_l_logits_sel is not None and l_indices is not None:
                    max_idx = ref_l_logits_sel.shape[-1] - 1
                    l_indices_safe = torch.clamp(l_indices, min=0, max=max_idx)
                    ref_l_log_probs = _gather_log_probs(ref_l_logits_sel, l_indices_safe)
                    if torch.isnan(ref_l_log_probs).any() or torch.isinf(ref_l_log_probs).any():
                        print(f"Warning: NaN/inf in ref_l_log_probs. Replacing with zeros.")
                        ref_l_log_probs = torch.zeros_like(ref_l_log_probs)
                else:
                    ref_l_log_probs = torch.zeros(num_patterns, device=config.device)
                
                ref_pattern_log_probs = (
                    ref_slot_log_probs.sum(dim=-1) + ref_h_log_probs + ref_l_log_probs
                )
                
                # Final check on ref_pattern_log_probs
                if torch.isnan(ref_pattern_log_probs).any() or torch.isinf(ref_pattern_log_probs).any():
                    print(f"Warning: NaN/inf in ref_pattern_log_probs after sum. Replacing with zeros.")
                    ref_pattern_log_probs = torch.zeros(num_patterns, device=config.device)
            except Exception as e:
                print(f"Warning: Error computing ref_pattern_log_probs: {e}. Using zeros.")
                ref_pattern_log_probs = torch.zeros(num_patterns, device=config.device)
    else:
        # PPO-style: use stored old_log_probs (from when we sampled)
        ref_pattern_log_probs = old_log_probs
        ref_aug_logits_sel = None
        ref_h_logits_sel = None
        ref_l_logits_sel = None
        if train_state.step == 1 or train_state.step % 100 == 0:
            print(f"Using PPO-style GRPO (old_log_probs), step={train_state.step}")
    
    # Safety check: ensure log_probs are finite before computing ratios
    if torch.isnan(pattern_log_probs).any() or torch.isinf(pattern_log_probs).any():
        print(f"Warning: NaN/inf in pattern_log_probs. Replacing with zeros.")
        pattern_log_probs = torch.zeros_like(pattern_log_probs)
    if torch.isnan(ref_pattern_log_probs).any() or torch.isinf(ref_pattern_log_probs).any():
        print(f"Warning: NaN/inf in ref_pattern_log_probs. Replacing with zeros.")
        ref_pattern_log_probs = torch.zeros_like(ref_pattern_log_probs)

    # Ratios vs reference policy for PPO-style surrogate
    advantages = rewards_tensor - train_state.baseline_reward  # [num_patterns]

    # Standard PPO advantage normalization: (adv - mean) / (std + eps)
    # This centers advantages around 0 and scales them to unit variance, which is critical for PPO stability
    if len(advantages) > 1:
        advantages_mean = advantages.mean()
        advantages_std = advantages.std()
        eps = 1e-8
        if advantages_std > eps:
            advantages = (advantages - advantages_mean) / (advantages_std + eps)
        else:
            # If std is too small, just center
            advantages = advantages - advantages_mean
    else:
        # Single pattern: just center it
        advantages = advantages - advantages.mean()

    # Clamp advantages after normalization (should be ~[-3, 3] after normalization, clamp to [-5, 5] for safety)
    advantages = torch.clamp(advantages, min=-5.0, max=5.0)

    # Compute ratio with proper clamping to keep ratios in reasonable range
    log_prob_diff = pattern_log_probs - ref_pattern_log_probs.detach()
    # Clamp log_prob_diff to keep ratios in reasonable range [exp(-2), exp(2)] ≈ [0.14, 7.4]
    # This is much tighter than before and prevents ratio explosion
    log_prob_diff = torch.clamp(log_prob_diff, min=-2.0, max=2.0)
    
    # Check for NaN/inf in log_probs and handle gracefully
    if torch.isnan(log_prob_diff).any() or torch.isinf(log_prob_diff).any():
        # If we have invalid log_probs, use a safe fallback: ratio = 1.0 (no update)
        print(f"Warning: NaN/inf detected in log_prob_diff. Using safe fallback.")
        ratio = torch.ones_like(pattern_log_probs)
    else:
        ratio = torch.exp(log_prob_diff)
    
    # Clamp ratio to reasonable range [0.1, 10] (much tighter than [1e-8, 1e8] to prevent huge losses)
    ratio = torch.clamp(ratio, min=0.1, max=10.0)
    
    # PPO clipped surrogate loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - config.ppo_clip_epsilon, 1 + config.ppo_clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Final safety check: if policy_loss is NaN/inf, use a small dummy loss
    if torch.isnan(policy_loss) or torch.isinf(policy_loss):
        print(f"Warning: policy_loss is NaN/inf. Using safe fallback loss.")
        policy_loss = torch.tensor(0.0, device=config.device, requires_grad=True)

    # Clip fraction for logging
    clip_frac = (ratio < (1 - config.ppo_clip_epsilon)).float().mean()
    clip_frac += (ratio > (1 + config.ppo_clip_epsilon)).float().mean()
    clip_frac = clip_frac.item() / 2.0

    # KL penalty to reference policy (GRPO anchor) - only if using reference model
    if should_use_reference and config.use_kl_penalty:
        def _categorical_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
            # Clamp logits to prevent numerical issues
            p_logits_safe = torch.clamp(p_logits.float(), min=-50.0, max=50.0)
            q_logits_safe = torch.clamp(q_logits.float(), min=-50.0, max=50.0)
            p_log = torch.log_softmax(p_logits_safe, dim=-1)
            q_log = torch.log_softmax(q_logits_safe, dim=-1)
            p_prob = torch.exp(p_log).clamp(min=1e-8, max=1.0)
            kl = (p_prob * (p_log - q_log)).sum(dim=-1)
            # Clamp KL to reasonable range
            kl = torch.clamp(kl, min=0.0, max=100.0)
            return kl

        kl_terms = []
        if (
            aug_logits is not None
            and ref_aug_logits_sel is not None
            and aug_logits.numel() > 0
        ):
            kl_terms.append(_categorical_kl(aug_logits, ref_aug_logits_sel).mean())
        if (
            h_logits is not None
            and ref_h_logits_sel is not None
            and h_logits.numel() > 0
        ):
            kl_terms.append(_categorical_kl(h_logits, ref_h_logits_sel).mean())
        if (
            l_logits is not None
            and ref_l_logits_sel is not None
            and l_logits.numel() > 0
        ):
            kl_terms.append(_categorical_kl(l_logits, ref_l_logits_sel).mean())

        kl_value = torch.stack(kl_terms).mean() if len(kl_terms) > 0 else torch.tensor(0.0, device=config.device)
        kl_loss = train_state.kl_beta * kl_value
    else:
        # No KL penalty in PPO-style mode
        kl_value = torch.tensor(0.0, device=config.device)
        kl_loss = torch.tensor(0.0, device=config.device)

    # 5. Entropy regularization (encourage exploration)
    # Compute entropy from per-slot log_probs (augmentation choices only)
    # Clamp log_probs to prevent numerical issues
    slot_log_probs_safe = torch.clamp(slot_log_probs, min=-50.0, max=0.0)  # log_probs should be <= 0
    probs = torch.exp(slot_log_probs_safe).clamp(min=1e-8, max=1.0)  # [num_patterns, slots]
    entropy = -(probs * slot_log_probs_safe).sum(dim=-1).mean()  # Average entropy over patterns
    # Ensure entropy is finite
    if torch.isnan(entropy) or torch.isinf(entropy):
        entropy = torch.tensor(0.0, device=config.device)
    entropy_loss = -config.entropy_coefficient * entropy  # Negative because we want to maximize entropy

    total_loss = policy_loss + entropy_loss + kl_loss
    
    # Final safety check: ensure total_loss is finite
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"Warning: total_loss is NaN/inf. Using safe fallback loss.")
        total_loss = torch.tensor(0.0, device=config.device, requires_grad=True)

    # 6. Backprop and update meta model
    train_state.meta_optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(train_state.meta_model.parameters(), max_norm=1.0)
    train_state.meta_optimizer.step()

    # 6b. Adaptive KL scaling (optional, only if using reference model)
    if should_use_reference and config.use_kl_penalty:
        high_thresh = config.kl_target * 1.5
        low_thresh = config.kl_target / 1.5
        if kl_value.item() > high_thresh:
            train_state.kl_beta *= config.kl_adaptation_rate
        elif kl_value.item() < low_thresh:
            train_state.kl_beta /= config.kl_adaptation_rate

    # 6c. Update reference model (EMA or periodic hard copy) - only if using reference model
    if should_use_reference and train_state.meta_ref_model is not None:
        if config.use_ref_ema:
            decay = config.ref_ema_decay
            with torch.no_grad():
                for ref_p, p in zip(train_state.meta_ref_model.parameters(), train_state.meta_model.parameters()):
                    ref_p.data.mul_(decay).add_(p.data, alpha=1 - decay)
        elif train_state.step % max(config.ref_update_interval, 1) == 0:
            train_state.meta_ref_model.load_state_dict(train_state.meta_model.state_dict())
    
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
        "meta/ratio_mean": ratio.mean().item(),
        "meta/ratio_std": ratio.std().item(),
        "meta/clip_fraction": clip_frac,
        "meta/kl_value": kl_value.item(),
        "meta/kl_beta": train_state.kl_beta,
        "meta/mean_reward": mean_reward,
        "meta/baseline_reward": train_state.baseline_reward,
        "meta/baseline_loss": baseline_loss,
        "meta/rewards": rewards,  # List of rewards
        "meta/step": train_state.step,
        "meta/using_reference_model": float(should_use_reference),  # Track which mode we're using
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


def evaluate_meta(
    config: MetaTrainConfig,
    train_state: MetaTrainState,
    base_model: TinyRecursiveReasoningModel_ACTV1,
    eval_loader: DataLoader,
    num_batches: int = 100,
) -> Dict[str, float]:
    """
    Evaluate MetaTRM without training (eval-only mode).
    
    Runs inference over multiple batches and aggregates metrics for stable evaluation.
    Similar to evaluate() in pretrain.py but for meta-training setup.
    
    Args:
        config: MetaTrainConfig
        train_state: MetaTrainState (meta model will be set to eval mode)
        base_model: Base TRM model
        eval_loader: DataLoader for evaluation batches
        num_batches: Number of batches to evaluate over
    
    Returns:
        Dictionary of aggregated metrics including:
        - eval/mean_reward: Mean reward across all batches
        - eval/baseline_loss: Mean baseline loss
        - eval/mean_meta_sample_time: Average time to sample patterns from meta model (seconds)
        - eval/mean_finetune_time: Average time for fine-tuning + evaluation (seconds)
        - eval/mean_total_time_per_batch: Average total time per batch (seconds)
        - trm_eval/baseline/*: Baseline TRM metrics
        - trm_eval/post/*_mean: Post-augmentation TRM metrics
    """
    train_state.meta_model.eval()
    base_model.eval()
    
    all_baseline_metrics = []
    all_pattern_metrics = []
    all_rewards = []
    all_baseline_losses = []
    
    # Timing metrics
    all_meta_sample_times = []
    all_finetune_times = []
    all_total_times = []
    
    eval_batch_count = 0
    val_iter = iter(eval_loader)
    
    # Initialize metrics buffer for aggregation (reset for each evaluation run)
    metrics_buffer = []
    # Aggregate metrics every config.eval_log_step batches before logging to wandb.
    # This controls how dense the eval curves are (similar to TRM's "every 10 steps").
    log_interval = max(config.eval_log_step, 1)
    
    # Define aggregation function (needed for both wandb logging and final aggregation)
    def aggregate_metrics_with_count(metrics_list, key, count_key="count"):
        """
        Aggregate metrics by accumulating counts (like pretrain.py).
        For exact_accuracy and similar ratio metrics, convert back to counts, sum, then normalize.
        """
        if len(metrics_list) == 0:
            return None
        
        # Special handling for ratio metrics that should be aggregated by count
        ratio_metrics = {"exact_accuracy", "accuracy", "q_halt_accuracy"}
        
        if key in ratio_metrics:
            # For ratio metrics: convert to counts, sum, then normalize
            total_numerator = 0.0
            total_denominator = 0.0
            
            for m in metrics_list:
                if key in m and count_key in m:
                    ratio = m[key]
                    count = m[count_key]
                    if ratio is not None and count is not None and count > 0:
                        # Convert ratio back to count of correct items
                        numerator = ratio * count
                        total_numerator += numerator
                        total_denominator += count
            
            if total_denominator > 0:
                return float(total_numerator / total_denominator)
            return None
        else:
            # For other metrics (loss, steps, etc.), use weighted average by count
            total_weighted_sum = 0.0
            total_weight = 0.0
            
            for m in metrics_list:
                if key in m and count_key in m:
                    value = m[key]
                    count = m[count_key]
                    if value is not None and count is not None and count > 0:
                        total_weighted_sum += value * count
                        total_weight += count
            
            if total_weight > 0:
                return float(total_weighted_sum / total_weight)
            return None
    
    print(f"Running evaluation over {num_batches} batches...")
    
    # Note: We don't use torch.inference_mode() here because compute_rewards_from_augmentations
    # needs to fine-tune the base model, which requires gradients. We only disable gradients
    # for meta model sampling.
    while eval_batch_count < num_batches:
        batch_start_time = time.time()
        try:
            _, val_batch, _ = next(val_iter)
        except StopIteration:
            # Reset iterator if we run out
            val_iter = iter(eval_loader)
            _, val_batch, _ = next(val_iter)
        
        # Move to device
        val_batch = {k: v.to(config.device) for k, v in val_batch.items()}
        
        # For eval-only mode:
        # - If few-shot is enabled, eval_loader contains train batches with few-shot data
        # - Otherwise, eval_loader contains val batches
        # We use the batch as both train and val batch for evaluation
        train_batch = val_batch
        
        # Handle few-shot batches if needed
        if config.use_few_shot and "similar_inputs" in train_batch:
            meta_batch = {
                "inputs": train_batch["inputs"],
                "labels": train_batch["labels"],
                "puzzle_identifiers": train_batch["puzzle_identifiers"],
            }
        else:
            meta_batch = train_batch
        
        # Sample patterns from meta model (no gradients needed for meta model)
        meta_sample_start = time.time()
        with torch.no_grad():
            (
                patterns,
                slot_log_probs,
                h_values,
                l_values,
                pattern_log_probs,
                sampled_indices,
                h_indices,
                l_indices,
                aug_logits,
                h_logits,
                l_logits,
            ) = sample_patterns_from_meta_model(
                meta_model=train_state.meta_model,
                batch=meta_batch,
                num_patterns=config.num_patterns_per_batch,
                temperature=1.0,
            )
        if config.device == "cuda":
            torch.cuda.synchronize()  # Ensure GPU operations complete
        meta_sample_time = time.time() - meta_sample_start
        all_meta_sample_times.append(meta_sample_time)
        
        # Prepare few-shot batch if needed
        few_shot_train_batch = None
        if config.use_few_shot and "similar_inputs" in train_batch:
            similar_inputs = train_batch["similar_inputs"]
            similar_labels = train_batch["similar_labels"]
            batch_size = similar_inputs.shape[0]
            num_similar = similar_inputs.shape[1]
            
            if num_similar > 0 and similar_inputs.numel() > 0:
                similar_inputs_flat = similar_inputs.reshape(-1, similar_inputs.shape[-1])
                similar_labels_flat = similar_labels.reshape(-1, similar_labels.shape[-1])
                
                if "similar_puzzle_identifiers" in train_batch:
                    similar_puzzle_ids_flat = train_batch["similar_puzzle_identifiers"].reshape(-1)
                else:
                    similar_puzzle_ids_flat = train_batch["puzzle_identifiers"].repeat_interleave(num_similar, dim=0)
                
                max_puzzle_id = base_model.config.num_puzzle_identifiers - 1
                if similar_puzzle_ids_flat.min() >= 0 and similar_puzzle_ids_flat.max() <= max_puzzle_id:
                    few_shot_train_batch = {
                        "inputs": similar_inputs_flat,
                        "labels": similar_labels_flat,
                        "puzzle_identifiers": similar_puzzle_ids_flat,
                    }
        
        # Compute rewards (this fine-tunes base model, needs gradients enabled!)
        finetune_start = time.time()
        rewards, baseline_loss, baseline_metrics, pattern_metrics = compute_rewards_from_augmentations(
            base_model=base_model,
            original_batch=meta_batch,  # Use same input for baseline and evaluation (matches pattern sampling)
            patterns=patterns,
            num_finetune_steps=config.num_finetune_steps,
            baseline_loss=None,
            use_binary_reward=config.use_binary_reward,
            reward_scale=config.reward_scale,
            loss_type=config.loss_type,
            grid_height=config.grid_height,
            grid_width=config.grid_width,
            few_shot_train_batch=few_shot_train_batch,
            meta_model=train_state.meta_model if config.use_few_shot else None,
            h_cycles=h_values,
            l_cycles=l_values,
            hl_cost_lambda=config.hl_cost_lambda,
            hl_cost_alpha=config.hl_cost_alpha,
            hl_cost_beta=config.hl_cost_beta,
        )
        if config.device == "cuda":
            torch.cuda.synchronize()  # Ensure GPU operations complete
        finetune_time = time.time() - finetune_start
        all_finetune_times.append(finetune_time)
        
        # Collect metrics
        all_baseline_metrics.append(baseline_metrics)
        all_pattern_metrics.extend(pattern_metrics)
        all_rewards.extend(rewards)
        all_baseline_losses.append(baseline_loss)
        
        batch_total_time = time.time() - batch_start_time
        all_total_times.append(batch_total_time)
        
        eval_batch_count += 1
        
        # Print timing for each batch (similar to TRM's "Completed inference in X steps")
        print(f"  Batch {eval_batch_count}/{num_batches} completed in {batch_total_time:.3f}s "
              f"(meta: {meta_sample_time:.3f}s, finetune: {finetune_time:.3f}s)")
        
        # DIAGNOSTIC: Print baseline and per-pattern metrics
        baseline_acc = baseline_metrics.get("exact_accuracy", 0.0)
        baseline_loss_val = baseline_loss
        
        print(f"    Baseline: acc={baseline_acc:.4f}, loss={baseline_loss_val:.4f}")
        
        # Analyze each pattern's performance
        improved_patterns = []
        worsened_patterns = []
        for i, (pattern, pattern_metric) in enumerate(zip(patterns, pattern_metrics)):
            pattern_acc = pattern_metric.get("exact_accuracy", 0.0)
            pattern_loss = pattern_metric.get("loss", float('inf'))
            reward_val = rewards[i] if i < len(rewards) else 0.0
            improvement = pattern_acc - baseline_acc
            loss_improvement = baseline_loss_val - pattern_loss  # Positive means loss decreased (better)
            
            # Get H_cycle and L_cycle for this pattern
            h_cycle = h_values[i] if i < len(h_values) else None
            l_cycle = l_values[i] if i < len(l_values) else None
            
            status = "✓" if improvement > 0 else "✗"
            cycle_info = f"H={h_cycle},L={l_cycle}" if h_cycle is not None and l_cycle is not None else ""
            print(f"    Pattern {i+1} [{pattern}] {cycle_info}: acc={pattern_acc:.4f} ({improvement:+.4f}), "
                  f"loss={pattern_loss:.4f} ({loss_improvement:+.4f}), reward={reward_val:.2f} {status}")
            
            if improvement > 0:
                improved_patterns.append((pattern, improvement, pattern_acc))
            elif improvement < 0:
                worsened_patterns.append((pattern, improvement, pattern_acc))
        
        # Summary
        if improved_patterns:
            print(f"    ✓ Improved patterns ({len(improved_patterns)}): {[p[0] for p in improved_patterns]}")
        if worsened_patterns:
            print(f"    ✗ Worsened patterns ({len(worsened_patterns)}): {[p[0] for p in worsened_patterns]}")
        
        # Compute mean post-augmentation accuracy
        post_accs = [pm.get("exact_accuracy", 0.0) for pm in pattern_metrics]
        mean_post_acc = sum(post_accs) / len(post_accs) if len(post_accs) > 0 else 0.0
        overall_improvement = mean_post_acc - baseline_acc
        print(f"    Overall: mean_post_acc={mean_post_acc:.4f}, improvement={overall_improvement:+.4f}")
        
        # Store metrics in buffer for aggregation (don't log every batch)
        batch_metrics_buffer = {
            "baseline_metrics": baseline_metrics,  # Has "count" field
            "pattern_metrics": pattern_metrics,     # List of dicts, each has "count"
            "rewards": rewards,
            "baseline_loss": baseline_loss,
            "meta_sample_time": meta_sample_time,
            "finetune_time": finetune_time,
            "total_time": batch_total_time,
        }
        metrics_buffer.append(batch_metrics_buffer)
        
        # Log to wandb every log_interval batches (aggregated)
        if eval_batch_count % log_interval == 0 and len(metrics_buffer) > 0:
            # Aggregate metrics from buffer using count-weighted method
            buffer_baseline = [m["baseline_metrics"] for m in metrics_buffer]
            buffer_patterns = []
            for m in metrics_buffer:
                buffer_patterns.extend(m["pattern_metrics"])
            
            # Define metric keys (same as used in final aggregation)
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
            
            # Aggregate baseline metrics
            aggregated_baseline = {}
            for key in trm_metric_keys:
                val = aggregate_metrics_with_count(buffer_baseline, key)
                if val is not None:
                    aggregated_baseline[f"trm_eval/baseline/{key}"] = val
            
            # Aggregate post-augmentation metrics
            aggregated_post = {}
            for key in trm_metric_keys:
                val = aggregate_metrics_with_count(buffer_patterns, key)
                if val is not None:
                    aggregated_post[f"trm_eval/post/{key}_mean"] = val
            
            # Aggregate other metrics
            buffer_rewards = []
            buffer_baseline_losses = []
            buffer_meta_times = []
            buffer_finetune_times = []
            buffer_total_times = []
            
            for m in metrics_buffer:
                buffer_rewards.extend(m["rewards"])
                buffer_baseline_losses.append(m["baseline_loss"])
                buffer_meta_times.append(m["meta_sample_time"])
                buffer_finetune_times.append(m["finetune_time"])
                buffer_total_times.append(m["total_time"])
            
            # Log aggregated metrics to wandb
            wandb_step = (eval_batch_count // log_interval) - 1  # 0-indexed
            wandb_metrics = {
                "eval/aggregated_reward": float(sum(buffer_rewards) / len(buffer_rewards)) if len(buffer_rewards) > 0 else 0.0,
                "eval/aggregated_baseline_loss": float(sum(buffer_baseline_losses) / len(buffer_baseline_losses)) if len(buffer_baseline_losses) > 0 else 0.0,
                "eval/aggregated_meta_sample_time": float(sum(buffer_meta_times) / len(buffer_meta_times)) if len(buffer_meta_times) > 0 else 0.0,
                "eval/aggregated_finetune_time": float(sum(buffer_finetune_times) / len(buffer_finetune_times)) if len(buffer_finetune_times) > 0 else 0.0,
                "eval/aggregated_total_time": float(sum(buffer_total_times) / len(buffer_total_times)) if len(buffer_total_times) > 0 else 0.0,
                "eval/batches_in_window": len(metrics_buffer),
                **aggregated_baseline,
                **aggregated_post,
            }
            
            wandb.log(wandb_metrics, step=wandb_step)
            
            # Clear buffer
            metrics_buffer = []
            
            print(f"  Logged aggregated metrics to wandb (batches {eval_batch_count - log_interval + 1}-{eval_batch_count})")
        
        # Print intermediate stats every 20 batches
        if eval_batch_count % 20 == 0:
            current_mean_reward = sum(all_rewards) / len(all_rewards) if len(all_rewards) > 0 else 0.0
            current_mean_loss = sum(all_baseline_losses) / len(all_baseline_losses) if len(all_baseline_losses) > 0 else 0.0
            avg_time = sum(all_total_times[-20:]) / min(20, len(all_total_times))
            print(f"  Progress: {eval_batch_count}/{num_batches} batches (avg time: {avg_time:.3f}s/batch)")
            print(f"  Intermediate stats (after {eval_batch_count} batches):")
            print(f"    Mean reward so far: {current_mean_reward:.4f}")
            print(f"    Mean baseline loss: {current_mean_loss:.4f}")
        elif eval_batch_count % 10 == 0:
            avg_time = sum(all_total_times[-10:]) / min(10, len(all_total_times))
            print(f"  Progress: {eval_batch_count}/{num_batches} batches (avg time: {avg_time:.3f}s/batch)")
        
    # Flush remaining metrics in buffer (if any)
    if len(metrics_buffer) > 0:
        # Aggregate metrics from remaining buffer
        buffer_baseline = [m["baseline_metrics"] for m in metrics_buffer]
        buffer_patterns = []
        for m in metrics_buffer:
            buffer_patterns.extend(m["pattern_metrics"])
        
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
        
        aggregated_baseline = {}
        for key in trm_metric_keys:
            val = aggregate_metrics_with_count(buffer_baseline, key)
            if val is not None:
                aggregated_baseline[f"trm_eval/baseline/{key}"] = val
        
        aggregated_post = {}
        for key in trm_metric_keys:
            val = aggregate_metrics_with_count(buffer_patterns, key)
            if val is not None:
                aggregated_post[f"trm_eval/post/{key}_mean"] = val
        
        buffer_rewards = []
        buffer_baseline_losses = []
        buffer_meta_times = []
        buffer_finetune_times = []
        buffer_total_times = []
        
        for m in metrics_buffer:
            buffer_rewards.extend(m["rewards"])
            buffer_baseline_losses.append(m["baseline_loss"])
            buffer_meta_times.append(m["meta_sample_time"])
            buffer_finetune_times.append(m["finetune_time"])
            buffer_total_times.append(m["total_time"])
        
        # Log remaining aggregated metrics to wandb
        wandb_step = eval_batch_count // log_interval  # Next step after last full window
        wandb_metrics = {
            "eval/aggregated_reward": float(sum(buffer_rewards) / len(buffer_rewards)) if len(buffer_rewards) > 0 else 0.0,
            "eval/aggregated_baseline_loss": float(sum(buffer_baseline_losses) / len(buffer_baseline_losses)) if len(buffer_baseline_losses) > 0 else 0.0,
            "eval/aggregated_meta_sample_time": float(sum(buffer_meta_times) / len(buffer_meta_times)) if len(buffer_meta_times) > 0 else 0.0,
            "eval/aggregated_finetune_time": float(sum(buffer_finetune_times) / len(buffer_finetune_times)) if len(buffer_finetune_times) > 0 else 0.0,
            "eval/aggregated_total_time": float(sum(buffer_total_times) / len(buffer_total_times)) if len(buffer_total_times) > 0 else 0.0,
            "eval/batches_in_window": len(metrics_buffer),
            **aggregated_baseline,
            **aggregated_post,
        }
        
        wandb.log(wandb_metrics, step=wandb_step)
        print(f"  Logged remaining aggregated metrics to wandb (batches {eval_batch_count - len(metrics_buffer) + 1}-{eval_batch_count})")
    
    # Aggregate metrics like pretrain.py: accumulate counts, then normalize
    # This gives stable metrics instead of averaging volatile per-batch ratios.
    # 
    # Key difference:
    # - OLD (volatile): exact_accuracy = mean([batch1_acc, batch2_acc, ...])
    # - NEW (stable): exact_accuracy = sum(correct_sequences) / sum(total_sequences)
    #
    # This matches pretrain.py's evaluate() function which accumulates metrics
    # across all batches before normalizing, resulting in smooth growth curves.
    
    # Aggregate baseline metrics
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
    
    aggregated_baseline = {}
    for key in trm_metric_keys:
        val = aggregate_metrics_with_count(all_baseline_metrics, key)
        if val is not None:
            aggregated_baseline[f"trm_eval/baseline/{key}"] = val
    
    # Aggregate post-augmentation metrics (same approach)
    aggregated_post = {}
    for key in trm_metric_keys:
        val = aggregate_metrics_with_count(all_pattern_metrics, key)
        if val is not None:
            aggregated_post[f"trm_eval/post/{key}_mean"] = val
    
    # Aggregate rewards
    mean_reward = float(sum(all_rewards) / len(all_rewards)) if len(all_rewards) > 0 else 0.0
    mean_baseline_loss = float(sum(all_baseline_losses) / len(all_baseline_losses)) if len(all_baseline_losses) > 0 else 0.0
    
    # Aggregate timing metrics
    mean_meta_sample_time = float(sum(all_meta_sample_times) / len(all_meta_sample_times)) if len(all_meta_sample_times) > 0 else 0.0
    mean_finetune_time = float(sum(all_finetune_times) / len(all_finetune_times)) if len(all_finetune_times) > 0 else 0.0
    mean_total_time = float(sum(all_total_times) / len(all_total_times)) if len(all_total_times) > 0 else 0.0
    
    metrics = {
        "eval/mean_reward": mean_reward,
        "eval/baseline_loss": mean_baseline_loss,
        "eval/num_batches": eval_batch_count,
        "eval/mean_meta_sample_time": mean_meta_sample_time,
        "eval/mean_finetune_time": mean_finetune_time,
        "eval/mean_total_time_per_batch": mean_total_time,
        **aggregated_baseline,
        **aggregated_post,
    }
    
    return metrics


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
    
    # Load vector database and setup few-shot if enabled
    vector_db = None
    meta_model_for_fewshot = None
    
    if config.use_few_shot:
        if config.vector_db_path and os.path.exists(config.vector_db_path):
            print(f"Loading vector database from {config.vector_db_path}")
            vector_db = PuzzleVectorDB.load(config.vector_db_path)
            print(f"Loaded vector database with {len(vector_db)} puzzles")
        else:
            raise ValueError(f"Vector database not found at {config.vector_db_path}. Please build it first using utils/build_vector_db.py")
    
    # Create base dataset first to get metadata
    base_train_dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths,
        global_batch_size=config.global_batch_size,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    ), split="train")
    train_metadata = base_train_dataset.metadata
    
    # Initialize training state (MetaTRM)
    train_state = init_meta_train_state(config, train_metadata)
    
    # Load base model
    base_model = load_base_model(config, train_metadata)
    base_model = base_model.to(device)
    
    # Extract base TRM model for few-shot (unwrap from loss head if needed)
    base_model_for_fewshot = base_model
    
    # Use the MetaTRM being trained for few-shot augmentation selection
    if config.use_few_shot:
        meta_model_for_fewshot = train_state.meta_model
    
    # Create dataloaders (with few-shot wrapper if enabled)
    train_loader, _ = create_dataloader(
        config,
        "train",
        rank=0,
        num_replicas=1,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        vector_db=vector_db,
        meta_model=meta_model_for_fewshot,
        base_model=base_model_for_fewshot,
    )
    
    val_loader = None
    val_metadata = None
    if len(config.data_paths_val) > 0:
        try:
            val_loader, val_metadata = create_dataloader(
                config, "val", rank=0, num_replicas=1, epochs_per_iter=1, global_batch_size=config.global_batch_size,
                vector_db=vector_db if config.use_few_shot else None,
                meta_model=meta_model_for_fewshot if config.use_few_shot else None,
                base_model=base_model_for_fewshot if config.use_few_shot else None,
            )
        except:
            print("Warning: Could not create validation loader, using train data for validation")
            val_loader, val_metadata = train_loader, train_metadata
    elif len(config.data_paths_test) > 0:
        try:
            val_loader, val_metadata = create_dataloader(
                config, "test", rank=0, num_replicas=1, epochs_per_iter=1, global_batch_size=config.global_batch_size,
                vector_db=vector_db if config.use_few_shot else None,
                meta_model=meta_model_for_fewshot if config.use_few_shot else None,
                base_model=base_model_for_fewshot if config.use_few_shot else None,
            )
        except:
            print("Warning: Could not create test loader, using train data for validation")
            val_loader, val_metadata = train_loader, train_metadata
    else:
        val_loader, val_metadata = train_loader, train_metadata
    
    # Eval loader choice: prefer val, else test (handled above), else train
    eval_loader_for_eval = val_loader if val_loader is not None else train_loader
    
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
    
    # Eval-only mode: run evaluation without training
    if config.eval_only:
        print("="*60)
        print("Running in EVAL-ONLY mode (no training)")
        print("="*60)
        
        # For eval-only: prioritize val_loader (test data) even when few-shot is enabled
        # Few-shot now works on test/val splits too, so we can use test data for proper evaluation
        eval_loader_for_eval = None
        if val_loader is not None:
            if config.use_few_shot:
                print("Few-shot enabled: using test/validation loader for evaluation (with few-shot support)")
            else:
                print("Using validation loader for evaluation")
            eval_loader_for_eval = val_loader
        else:
            if config.use_few_shot:
                print("Warning: No validation/test loader available. Using train loader for evaluation (with few-shot support).")
            else:
                print("Warning: No validation loader available. Using train loader for evaluation.")
            eval_loader_for_eval = train_loader
        
        eval_metrics = evaluate_meta(
            config=config,
            train_state=train_state,
            base_model=base_model,
            eval_loader=eval_loader_for_eval,
            num_batches=config.eval_num_batches,
        )
        
        # Print results
        print("\n" + "="*60)
        print("EVAL-ONLY RESULTS (averaged over batches)")
        print("="*60)
        print(f"Mean reward: {eval_metrics.get('eval/mean_reward', 0.0):.4f}")
        print(f"Baseline loss: {eval_metrics.get('eval/baseline_loss', 0.0):.4f}")
        print(f"Number of batches evaluated: {eval_metrics.get('eval/num_batches', 0)}")
        print("\nBaseline Metrics:")
        for key in ["loss", "accuracy", "exact_accuracy", "q_halt_accuracy"]:
            full_key = f"trm_eval/baseline/{key}"
            if full_key in eval_metrics:
                print(f"  {key}: {eval_metrics[full_key]:.4f}")
        print("\nPost-Augmentation Metrics (mean):")
        for key in ["loss", "accuracy", "exact_accuracy"]:
            full_key = f"trm_eval/post/{key}_mean"
            if full_key in eval_metrics:
                print(f"  {key}: {eval_metrics[full_key]:.4f}")
        print("="*60)
        
        # Log to wandb
        wandb.log(eval_metrics, step=0)
        wandb.finish()
        print("\nEval-only completed!")
        return
    
    # Training Loop
    print(f"Starting meta training for {config.meta_epochs} epochs...")
    
    for epoch in range(config.meta_epochs):
        print(f"\nEpoch {epoch + 1}/{config.meta_epochs}")
        
        train_state.meta_model.train()
        
        epoch_metrics = None
        
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
            
            # Keep track of last metrics for epoch-end logging
            epoch_metrics = metrics
            
            # Log training metrics every config.log_interval steps to reduce noise in W&B
            if train_state.step % max(config.log_interval, 1) == 0:
                wandb.log(metrics, step=train_state.step)
            
            # Update progress bar every step
            progress_bar.update(1)
            
            # Checkpoint
            if train_state.step % config.checkpoint_interval == 0:
                save_checkpoint(config, train_state)
        
        # Log metrics at end of epoch (if we have any)
        if epoch_metrics is not None:
            wandb.log(epoch_metrics, step=train_state.step)
        
        # Periodic full evaluation (like pretrain: every eval_interval epochs)
        if (epoch + 1) % config.eval_interval == 0:
            print(f"\nEVALUATE at epoch {epoch + 1}")
            eval_metrics = evaluate_meta(
                config=config,
                train_state=train_state,
                base_model=base_model,
                eval_loader=eval_loader_for_eval,
                num_batches=config.eval_num_batches,
            )
            wandb.log(eval_metrics, step=train_state.step)
        
        # Epoch checkpoint
        save_checkpoint(config, train_state)
    
    # Final checkpoint
    save_checkpoint(config, train_state)
    wandb.finish()
    print("Meta training completed!")


if __name__ == "__main__":
    launch()

