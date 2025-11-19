from __future__ import annotations

from typing import List, Tuple, Dict

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

