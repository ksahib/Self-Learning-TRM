from __future__ import annotations

from typing import List, Tuple, Dict, Optional, Callable

import torch
import numpy as np

from .augmentation import (
    POSITION_INDEX_TO_CHOICES,
    AUG_TYPE_TO_ID,
    apply_augmentation_sequence_to_whole_grid,
    flatten_grid_to_sequence,
    unflatten_sequence_to_grid,
)


PATTERN_LENGTH = 6


def _scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def greedy_indices_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits: [batch, slots, choices]
    Returns:
        indices: [batch, slots]
    """
    return torch.argmax(logits, dim=-1)


def sample_indices_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample augmentation indices and return their log probabilities.
    Args:
        logits: [batch, slots, choices]
        temperature: sampling temperature
    Returns:
        sampled_indices: [batch, slots]
        log_probs: [batch, slots]
    """
    scaled = _scale_logits(logits, temperature)
    probs = torch.softmax(scaled, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    probs_sum = probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    probs = probs / probs_sum

    batch, slots, choices = probs.shape
    flat = probs.view(batch * slots, choices)
    sampled = torch.multinomial(flat, num_samples=1).view(batch, slots)

    log_probs = torch.log(probs.clamp_min(1e-8))
    selected = log_probs.gather(dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)
    return sampled, selected


def indices_to_choice_chars(indices: torch.Tensor) -> List[List[str]]:
    """
    Map binary indices to augmentation characters per position.
    Args:
        indices: [batch, slots]
    Returns:
        chars: list over batch, each list contains PATTERN_LENGTH chars
    """
    batch, slots = indices.shape
    chars: List[List[str]] = []
    for b in range(batch):
        sample_chars: List[str] = []
        for slot in range(slots):
            idx = int(indices[b, slot].item())
            choices = POSITION_INDEX_TO_CHOICES[slot]
            idx = max(0, min(idx, len(choices) - 1))
            sample_chars.append(choices[idx])
        chars.append(sample_chars)
    return chars


def indices_to_pattern_strings(indices: torch.Tensor) -> List[str]:
    """
    Convert binary choices to column-major pattern strings.
    """
    char_lists = indices_to_choice_chars(indices)
    patterns = []
    for chars in char_lists:
        left_col = chars[:3]
        right_col = chars[3:]
        patterns.append("".join(left_col + right_col))
    return patterns


def indices_to_grid_tensor(indices: torch.Tensor) -> torch.Tensor:
    """
    Convert indices to augmentation ID grid [batch, 3, 2].
    """
    device = indices.device
    chars = indices_to_choice_chars(indices)
    batch = len(chars)
    grid = torch.empty((batch, 3, 2), dtype=torch.long, device=device)
    for b, char_list in enumerate(chars):
        for row in range(3):
            left_char = char_list[row]
            right_char = char_list[row + 3]
            grid[b, row, 0] = AUG_TYPE_TO_ID[left_char]
            grid[b, row, 1] = AUG_TYPE_TO_ID[right_char]
    return grid


def apply_augmentation_patterns_to_batch(
    batch: Dict[str, torch.Tensor],
    patterns: List[str],
    grid_height: int = 9,
    grid_width: int = 9,
) -> Dict[str, torch.Tensor]:
    """
    Apply augmentation patterns to a batch of puzzles.
    
    Each puzzle in the batch is augmented with its corresponding pattern.
    If there are more patterns than puzzles, patterns are cycled.
    If there are fewer patterns than puzzles, the last pattern is repeated.
    
    Args:
        batch: Dictionary with 'inputs', 'labels', and optionally other keys.
               'inputs' and 'labels' should be [batch_size, seq_len] tensors.
        patterns: List of augmentation pattern strings (e.g., ['LR.V.T', '.RH.OT']).
                  Each pattern is a 6-character string from MetaTRM output.
        grid_height: Height of the grid (default 9 for Sudoku).
        grid_width: Width of the grid (default 9 for Sudoku).
    
    Returns:
        Augmented batch with same structure as input, where each puzzle
        has been transformed according to its pattern.
    """
    batch_size = batch["inputs"].shape[0]
    device = batch["inputs"].device
    dtype = batch["inputs"].dtype
    
    # Convert to numpy for augmentation
    inputs_np = batch["inputs"].cpu().numpy()
    labels_np = batch["labels"].cpu().numpy() if "labels" in batch else None
    
    aug_inputs_list = []
    aug_labels_list = []
    
    for i in range(batch_size):
        # Get pattern for this puzzle (cycle if needed)
        pattern = patterns[i % len(patterns)]
        
        # Unflatten input grid
        input_grid = unflatten_sequence_to_grid(inputs_np[i], grid_height, grid_width)
        
        # Apply augmentation sequence to whole grid
        aug_input_grid = apply_augmentation_sequence_to_whole_grid(input_grid, pattern)
        
        # Flatten back
        aug_input_sequence = flatten_grid_to_sequence(aug_input_grid)
        aug_inputs_list.append(aug_input_sequence)
        
        # Apply same augmentation to labels if present
        if labels_np is not None:
            label_grid = unflatten_sequence_to_grid(labels_np[i], grid_height, grid_width)
            aug_label_grid = apply_augmentation_sequence_to_whole_grid(label_grid, pattern)
            aug_label_sequence = flatten_grid_to_sequence(aug_label_grid)
            aug_labels_list.append(aug_label_sequence)
    
    # Convert back to tensors
    aug_inputs = torch.tensor(
        np.stack(aug_inputs_list), dtype=dtype, device=device
    )
    
    # Build augmented batch
    aug_batch = {"inputs": aug_inputs}
    
    if labels_np is not None:
        aug_labels = torch.tensor(
            np.stack(aug_labels_list), dtype=batch["labels"].dtype, device=device
        )
        aug_batch["labels"] = aug_labels
    
    # Copy other keys (like puzzle_identifiers) if present
    for key in batch:
        if key not in ("inputs", "labels"):
            aug_batch[key] = batch[key]
    
    return aug_batch


def meta_trm_to_base_trm_pipeline(
    meta_model: "MetaTRM",  # Forward reference to avoid circular import
    base_model: "TinyRecursiveReasoningModel_ACTV1",  # Forward reference
    batch: Dict[str, torch.Tensor],
    sample: bool = True,
    temperature: float = 1.0,
    num_finetune_steps: int = 0,
    grid_height: int = 9,
    grid_width: int = 9,
) -> Tuple[List[str], torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Complete pipeline: MetaTRM samples augmentations → apply → base TRM processes.
    
    This function orchestrates the full flow:
    1. MetaTRM forward pass to sample augmentation patterns
    2. Apply augmentations to batch
    3. Run base TRM on each augmented batch
    4. Return patterns, log probabilities, and base TRM outputs
    
    Args:
        meta_model: MetaTRM instance
        base_model: Base TRM model instance
        batch: Input batch with 'inputs', 'labels', etc.
        sample: If True, sample augmentations; if False, use greedy
        temperature: Sampling temperature (only used if sample=True)
        num_finetune_steps: Number of fine-tuning steps for base TRM per augmentation
        grid_height: Height of grid (default 9 for Sudoku)
        grid_width: Width of grid (default 9 for Sudoku)
    
    Returns:
        patterns: List of sampled augmentation pattern strings
        log_probs: Tensor of log probabilities [batch, slots] for RL
        base_outputs: List of base TRM output dicts (one per pattern)
    """
    # Import here to avoid circular imports
    from models.recursive_reasoning.metaTRM import MetaTRM
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    
    # 1. MetaTRM forward pass
    meta_carry = meta_model.initial_carry(batch)
    meta_carry, meta_outputs = meta_model(
        meta_carry,
        batch,
        sample=sample,
        temperature=temperature,
        greedy=not sample,
    )
    
    # 2. Extract patterns and log probabilities
    if sample:
        patterns = meta_outputs["sampled_patterns"]
        log_probs = meta_outputs["sampled_log_probs"]  # [batch, slots]
    else:
        patterns = meta_outputs["greedy_patterns"]
        # For greedy, log_probs would be deterministic (not useful for RL)
        # Create dummy log_probs for consistency
        batch_size = batch["inputs"].shape[0]
        log_probs = torch.zeros(batch_size, PATTERN_LENGTH, device=batch["inputs"].device)
    
    # 3. Execute base TRM with augmentations
    base_outputs = MetaTRM.execute_base_trm_with_augmentations(
        base_model=base_model,
        batch=batch,
        patterns=patterns,
        num_finetune_steps=num_finetune_steps,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    
    return patterns, log_probs, base_outputs


def infer_vocab_size_from_checkpoint(checkpoint_path: str) -> int:
    """
    Infer vocab_size from checkpoint by examining embedding weight shapes.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Inferred vocab_size
    """
    import torch
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Try to find embedding weights
    embedding_keys = [
        "inner.embed_tokens.embedding_weight",
        "embed_tokens.embedding_weight",
        "_orig_mod.model.inner.embed_tokens.embedding_weight",
        "_orig_mod.inner.embed_tokens.embedding_weight",
    ]
    
    for key in embedding_keys:
        if key in state_dict:
            vocab_size = state_dict[key].shape[0]
            print(f"Inferred vocab_size={vocab_size} from checkpoint key: {key}")
            return vocab_size
    
    # Fallback: try lm_head weight
    lm_head_keys = [
        "inner.lm_head.weight",
        "lm_head.weight",
        "_orig_mod.model.inner.lm_head.weight",
        "_orig_mod.inner.lm_head.weight",
    ]
    
    for key in lm_head_keys:
        if key in state_dict:
            vocab_size = state_dict[key].shape[0]
            print(f"Inferred vocab_size={vocab_size} from checkpoint key: {key}")
            return vocab_size
    
    raise ValueError("Could not infer vocab_size from checkpoint")


def load_base_trm_checkpoint(
    checkpoint_path: str,
    model: "TinyRecursiveReasoningModel_ACTV1",  # Forward reference
    map_location: str = "cpu",
    strict: bool = False,
) -> "TinyRecursiveReasoningModel_ACTV1":
    """
    Load pretrained checkpoint into base TRM model.
    
    Handles:
    - torch.compile prefixes (_orig_mod.)
    - Missing/extra keys (if strict=False)
    - Puzzle embedding shape mismatches
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., "tryinghugs/step_21700")
        model: Base TRM model instance to load weights into
        map_location: Device to load checkpoint on ("cpu", "cuda", etc.)
        strict: If True, requires exact key match; if False, allows missing/extra keys
    
    Returns:
        Model with loaded weights (same instance, modified in-place)
    """
    import torch
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    
    # Handle torch.compile prefixes (_orig_mod.)
    # Check if state dict has _orig_mod prefix but model doesn't
    has_orig_mod = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    
    if has_orig_mod:
        # Remove _orig_mod prefix from keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod."):]
                # Also remove "model." prefix if present (from ACTLossHead wrapper)
                if new_key.startswith("model."):
                    new_key = new_key[len("model."):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    # Handle puzzle embedding shape mismatches (similar to pretrain.py)
    puzzle_emb_keys = [
        "inner.puzzle_emb.weights",
        "puzzle_emb.weights",
    ]
    
    for puzzle_emb_key in puzzle_emb_keys:
        if puzzle_emb_key in state_dict:
            if hasattr(model, "inner") and hasattr(model.inner, "puzzle_emb"):
                expected_shape = model.inner.puzzle_emb.weights.shape
                loaded_shape = state_dict[puzzle_emb_key].shape
                
                if loaded_shape != expected_shape:
                    print(
                        f"Puzzle embedding shape mismatch for {puzzle_emb_key}. "
                        f"Found {loaded_shape}, Expected {expected_shape}. "
                        f"Re-initializing using mean."
                    )
                    # Re-initialize using mean (similar to pretrain.py)
                    state_dict[puzzle_emb_key] = (
                        torch.mean(state_dict[puzzle_emb_key], dim=0, keepdim=True)
                        .expand(expected_shape)
                        .contiguous()
                    )
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"  - {key}")
        else:
            for key in missing_keys[:10]:
                print(f"  - {key}")
            print(f"  ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"  - {key}")
        else:
            for key in unexpected_keys[:10]:
                print(f"  - {key}")
            print(f"  ... and {len(unexpected_keys) - 10} more")
    
    print("Checkpoint loaded successfully!")
    return model


def get_lora_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """
    Extract only LoRA parameters (lora_A, lora_B) from model for optimizer.
    
    This is used to create an optimizer that only updates LoRA weights
    during base TRM fine-tuning, keeping the base model weights frozen.
    
    Args:
        model: PyTorch model (should contain LoRACastedLinear layers)
    
    Returns:
        List of LoRA parameters (lora_A and lora_B from all LoRA layers)
    """
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(param)
    return lora_params


def compute_base_trm_eval(
    base_model: "TinyRecursiveReasoningModel_ACTV1",
    batch: Dict[str, torch.Tensor],
    loss_fn: Optional[Callable] = None,
    loss_type: str = "softmax_cross_entropy",
) -> Dict[str, float]:
    """
    Run a forward pass (with ACT) on the base TRM and compute evaluation metrics.
    
    Returns:
        Dictionary with detached loss, token accuracy and sequence accuracy values.
    """
    from models.losses import IGNORE_LABEL_ID, softmax_cross_entropy, stablemax_cross_entropy
    
    if loss_fn is None:
        if loss_type == "stablemax_cross_entropy":
            loss_fn = stablemax_cross_entropy
        else:
            loss_fn = softmax_cross_entropy
    
    base_model.eval()
    
    with torch.inference_mode():
        carry = base_model.initial_carry(batch)
        all_finish = False
        while not all_finish:
            carry, outputs = base_model(carry=carry, batch=batch)
            all_finish = carry.halted.all().item()
        
        logits = outputs["logits"]
        labels = batch["labels"]
        
        mask = (labels != IGNORE_LABEL_ID)
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
        valid_sequences = carry.halted & (loss_counts > 0)
        valid_count = int(valid_sequences.sum().item())
        normalizer = max(valid_count, 1)
        
        if loss_type == "stablemax_cross_entropy":
            per_token_loss = loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)
        else:
            per_token_loss = loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID)
        
        lm_loss_sum = (per_token_loss / loss_divisor).sum()
        lm_loss_value = float((lm_loss_sum / normalizer).detach().cpu().item())
        
        preds = logits.argmax(dim=-1)
        correct_tokens = (preds == labels) & mask
        token_accuracy_sum = torch.where(
            valid_sequences,
            (correct_tokens.to(torch.float32) / loss_divisor).sum(-1),
            torch.zeros_like(loss_counts, dtype=torch.float32),
        ).sum()
        token_accuracy = float((token_accuracy_sum / normalizer).detach().cpu().item())
        
        seq_is_correct = correct_tokens.sum(-1) == loss_counts
        exact_accuracy_sum = (valid_sequences & seq_is_correct).sum()
        exact_accuracy = float(exact_accuracy_sum.item() / normalizer)
        
        q_halt_accuracy = None
        q_halt_loss_value = None
        if "q_halt_logits" in outputs:
            q_halt_logits = outputs["q_halt_logits"]
            q_halt_pred_correct = (q_halt_logits >= 0) == seq_is_correct
            q_halt_accuracy_sum = torch.where(
                valid_sequences, q_halt_pred_correct.to(torch.float32), torch.zeros_like(loss_counts, dtype=torch.float32)
            ).sum()
            q_halt_accuracy = float(q_halt_accuracy_sum.item() / normalizer)
            
            q_halt_loss_sum = torch.nn.functional.binary_cross_entropy_with_logits(
                q_halt_logits, seq_is_correct.to(q_halt_logits.dtype), reduction="sum"
            )
            q_halt_loss_value = float(q_halt_loss_sum.detach().cpu().item() / normalizer)
        
        q_continue_loss_value = None
        if "target_q_continue" in outputs and "q_continue_logits" in outputs:
            q_continue_loss_sum = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum"
            )
            q_continue_loss_value = float(q_continue_loss_sum.detach().cpu().item() / normalizer)
        
        steps_sum = torch.where(
            valid_sequences, carry.steps.to(torch.float32), torch.zeros_like(loss_counts, dtype=torch.float32)
        ).sum()
        avg_steps = float(steps_sum.item() / normalizer)
        
    total_tokens = int(mask.sum().item())
    total_sequences = int((loss_counts > 0).sum().item())
    
    return {
        "loss": lm_loss_value,
        "lm_loss": lm_loss_value,
        "token_accuracy": token_accuracy,
        "accuracy": token_accuracy,
        "sequence_accuracy": exact_accuracy,
        "exact_accuracy": exact_accuracy,
        "q_halt_accuracy": q_halt_accuracy,
        "q_halt_loss": q_halt_loss_value,
        "q_continue_loss": q_continue_loss_value,
        "steps": avg_steps,
        "count": valid_count,
        "total_tokens": total_tokens,
        "total_sequences": total_sequences,
    }


def compute_base_trm_loss(
    base_model: "TinyRecursiveReasoningModel_ACTV1",
    batch: Dict[str, torch.Tensor],
    loss_fn: Optional[Callable] = None,
    loss_type: str = "softmax_cross_entropy",
) -> torch.Tensor:
    """
    Backwards-compatible wrapper that returns only the detached loss tensor.
    """
    metrics = compute_base_trm_eval(
        base_model=base_model,
        batch=batch,
        loss_fn=loss_fn,
        loss_type=loss_type,
    )
    return torch.tensor(metrics["loss"])


def compute_reward(
    validation_loss: torch.Tensor,
    reward_scale: float = 1.0,
) -> float:
    """
    Compute reward from base TRM validation loss.
    
    Lower loss = higher reward (inverted relationship).
    This is used for REINFORCE policy gradient updates.
    
    Args:
        validation_loss: Detached validation loss tensor (scalar, no gradients)
        reward_scale: Scaling factor for reward (default 1.0)
    
    Returns:
        Reward value as Python float (not tensor)
    
    Example:
        If validation_loss = 0.5 and reward_scale = 1.0:
        reward = -0.5 * 1.0 = -0.5
        
        If validation_loss = 0.1 (better):
        reward = -0.1 * 1.0 = -0.1 (higher reward, less negative)
    """
    # Ensure loss is detached and on CPU
    if isinstance(validation_loss, torch.Tensor):
        loss_value = validation_loss.detach().cpu().item()
    else:
        loss_value = float(validation_loss)
    
    # Reward = -loss * scale
    # Lower loss → higher (less negative) reward
    reward = -loss_value * reward_scale
    
    return reward


def compute_rewards_from_augmentations(
    base_model: "TinyRecursiveReasoningModel_ACTV1",
    original_batch: Dict[str, torch.Tensor],
    patterns: List[str],
    num_finetune_steps: int,
    baseline_loss: Optional[float] = None,
    use_binary_reward: bool = True,
    reward_scale: float = 1.0,
    loss_type: str = "softmax_cross_entropy",
    grid_height: int = 9,
    grid_width: int = 9,
) -> Tuple[
    List[float],
    float,
    Dict[str, float],
    List[Dict[str, float]],
]:
    """
    Complete flow matching requ.txt: Fine-tune on augmented data → Evaluate on original input.
    
    For each augmentation pattern:
    1. Apply augmentation to original batch
    2. Fine-tune base TRM with LoRA on augmented batch
    3. Evaluate fine-tuned model on ORIGINAL (non-augmented) batch
    4. Compute binary reward: 1 if loss improved, 0 if not
    
    Args:
        base_model: Base TRM model (will be fine-tuned in-place, then restored)
        original_batch: Original batch with 'inputs', 'labels', etc. (used for evaluation)
        patterns: List of augmentation patterns (e.g., ['LR.V.T', '.RH.OT', 'LRH.VT'])
        num_finetune_steps: Number of LoRA fine-tuning steps per augmentation
        baseline_loss: Validation loss before any fine-tuning (if None, will compute)
        use_binary_reward: If True, binary reward (1 if improved, 0 if not)
                         If False, continuous reward = -loss * scale
        reward_scale: Scaling for continuous rewards
        loss_type: Type of loss function ('softmax_cross_entropy' or 'stablemax_cross_entropy')
        grid_height: Height of grid (default 9 for Sudoku)
        grid_width: Width of grid (default 9 for Sudoku)
    
    Returns:
        rewards: List of reward values (one per pattern), e.g., [0.0, 1.0, 0.0]
        baseline_loss: The baseline loss value used for comparison
        baseline_metrics: Dict with loss/accuracy stats before fine-tuning
        pattern_eval_metrics: List of dicts with loss/accuracy per pattern
    """
    import copy
    
    # 1. Compute baseline metrics (on original batch, before fine-tuning)
    base_model.eval()
    baseline_metrics = compute_base_trm_eval(
        base_model=base_model,
        batch=original_batch,
        loss_type=loss_type
    )
    if baseline_loss is None:
        baseline_loss = baseline_metrics["loss"]
    pattern_eval_metrics: List[Dict[str, float]] = []
    # 2. Save original model state (to restore after each fine-tuning)
    original_state = copy.deepcopy(base_model.state_dict())
    
    rewards = []
    
    # 3. For each augmentation pattern:
    for pattern in patterns:
        # 3a. Restore base model to original state
        base_model.load_state_dict(original_state)
        
        # 3b. Apply augmentation to batch (for fine-tuning)
        aug_batch = apply_augmentation_patterns_to_batch(
            original_batch, [pattern], grid_height=grid_height, grid_width=grid_width
        )
        
        # 3c. Fine-tune base TRM with LoRA on augmented data
        base_model.train()
        lora_params = get_lora_parameters(base_model)
        if len(lora_params) == 0:
            raise ValueError("No LoRA parameters found in model. Make sure model uses LoRACastedLinear layers.")
        
        optimizer = torch.optim.Adam(lora_params, lr=1e-4)
        
        for _ in range(num_finetune_steps):
            carry = base_model.initial_carry(aug_batch)
            carry, outputs = base_model(carry=carry, batch=aug_batch)
            
            if "labels" in aug_batch:
                logits = outputs["logits"]
                labels = aug_batch["labels"].long()  # Convert to int64 for CUDA cross_entropy
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        # 3d. Evaluate on ORIGINAL batch (not augmented!)
        base_model.eval()
        eval_metrics = compute_base_trm_eval(
            base_model=base_model,
            batch=original_batch,  # Original batch, not augmented
            loss_type=loss_type
        )
        val_loss = eval_metrics["loss"]
        pattern_eval_metrics.append(eval_metrics)
        
        # 3e. Compute reward
        if use_binary_reward:
            # Binary: 1 if improved, 0 if not
            reward = 1.0 if val_loss < baseline_loss else 0.0
        else:
            # Continuous: -loss * scale (lower loss = higher reward)
            reward = compute_reward(torch.tensor(val_loss), reward_scale=reward_scale)
        
        rewards.append(reward)
    
    # 4. Restore original model state
    base_model.load_state_dict(original_state)
    
    return rewards, baseline_loss, baseline_metrics, pattern_eval_metrics

