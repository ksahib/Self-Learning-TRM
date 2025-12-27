"""
Pre-train LoRAs for all architecture combinations.

This script:
1. Loads base TRM with minimal architecture (H_cycles=1, L_cycles=3)
2. For each (H, L) combination in h_cycle_choices × l_cycle_choices:
   - Creates model with that architecture (using minimal arch weights)
   - Freezes base weights, trains LoRA for N steps
   - Saves LoRA to lora_H{H}_L{L}.pth
"""

import os
import argparse
import torch
from typing import List, Dict, Tuple
import tqdm

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.recursive_reasoning.helper import (
    load_base_trm_checkpoint,
    get_lora_parameters,
    get_lora_state_dict,
    save_lora_state_dict,
)


def create_dataset(
    data_paths: List[str],
    batch_size: int,
    seed: int = 0,
) -> Tuple[PuzzleDataset, PuzzleDatasetMetadata]:
    """Create dataset for training."""
    # Verify data paths exist and have train/ subdirectory
    for data_path in data_paths:
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Data path does not exist: {data_path}\n"
                f"Expected structure: {data_path}/train/dataset.json"
            )
        train_path = os.path.join(data_path, "train", "dataset.json")
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Dataset structure not found: {train_path}\n"
                f"Expected: {data_path}/train/dataset.json\n"
                f"Please ensure your dataset has a 'train' subdirectory with 'dataset.json'"
            )
    
    dataset_config = PuzzleDatasetConfig(
        seed=seed,
        dataset_paths=data_paths,
        global_batch_size=batch_size,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    dataset = PuzzleDataset(dataset_config, split="train")
    metadata = dataset.metadata
    
    return dataset, metadata


def train_lora_for_architecture(
    base_model: TinyRecursiveReasoningModel_ACTV1,
    h_cycles: int,
    l_cycles: int,
    dataset: PuzzleDataset,
    num_steps: int,
    lr: float,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Train LoRA for a specific architecture.
    
    Args:
        base_model: Base TRM model (with minimal architecture weights)
        h_cycles: H_cycles for this architecture
        l_cycles: L_cycles for this architecture
        dataset: Training dataset (PuzzleDataset)
        num_steps: Number of training steps
        lr: Learning rate
        device: Device to train on
    
    Returns:
        LoRA state dict
    """
    # Update model config to use new architecture
    base_model.config.H_cycles = h_cycles
    base_model.config.L_cycles = l_cycles
    
    # Reset LoRA parameters to zero (start fresh for each architecture)
    for name, param in base_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.data.zero_()
    
    # Get LoRA parameters
    lora_params = get_lora_parameters(base_model)
    if len(lora_params) == 0:
        raise ValueError("No LoRA parameters found in model.")
    
    # Create optimizer
    optimizer = torch.optim.Adam(lora_params, lr=lr)
    
    # Training loop
    base_model.train()
    base_model = base_model.to(device)
    
    step = 0
    dataset_iter = iter(dataset)
    
    pbar = tqdm.tqdm(total=num_steps, desc=f"Training LoRA H={h_cycles} L={l_cycles}")
    
    while step < num_steps:
        try:
            # PuzzleDataset yields (set_name, batch, global_batch_size)
            set_name, batch, global_batch_size = next(dataset_iter)
        except StopIteration:
            dataset_iter = iter(dataset)
            set_name, batch, global_batch_size = next(dataset_iter)
        
        # Ensure batch is a dictionary
        if not isinstance(batch, dict):
            raise ValueError(f"Batch is not a dictionary. Type: {type(batch)}, Value: {batch}")
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        carry = base_model.initial_carry(batch)
        carry, outputs = base_model(carry=carry, batch=batch)
        
        # Compute loss
        if "labels" in batch:
            logits = outputs["logits"]
            labels = batch["labels"].long()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Extract LoRA state dict
    lora_state_dict = get_lora_state_dict(base_model)
    
    return lora_state_dict


def main():
    parser = argparse.ArgumentParser(description="Pre-train LoRAs for all architecture combinations")
    parser.add_argument("--base_checkpoint", type=str, required=True,
                       help="Path to base TRM checkpoint (trained on minimal architecture)")
    parser.add_argument("--lora_dir", type=str, default="loras/",
                       help="Directory to save LoRA files")
    parser.add_argument("--data_paths", type=str, nargs="+", required=True,
                       help="Paths to training data")
    parser.add_argument("--num_steps", type=int, default=1000,
                       help="Number of training steps per architecture")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate for LoRA training")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to train on")
    parser.add_argument("--h_cycle_choices", type=int, nargs="+", default=[1, 2, 3],
                       help="H_cycle choices")
    parser.add_argument("--l_cycle_choices", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                       help="L_cycle choices")
    parser.add_argument("--minimal_h_cycles", type=int, default=1,
                       help="Minimal H_cycles for base model")
    parser.add_argument("--minimal_l_cycles", type=int, default=3,
                       help="Minimal L_cycles for base model")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.lora_dir, exist_ok=True)
    
    # Load base model with minimal architecture
    print(f"Loading base model from {args.base_checkpoint}")
    print(f"Minimal architecture: H_cycles={args.minimal_h_cycles}, L_cycles={args.minimal_l_cycles}")
    
    # Create dataset to get metadata
    dataset, metadata = create_dataset(
        data_paths=args.data_paths,
        batch_size=args.batch_size,
    )
    
    # Create base model config with minimal architecture
    # We'll use a default config structure - you may need to adjust based on your checkpoint
    base_arch_cfg = {
        "H_cycles": args.minimal_h_cycles,
        "L_cycles": args.minimal_l_cycles,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4.0,
        "pos_encodings": "rope",
        "forward_dtype": "bfloat16",
        "mlp_t": False,
        "puzzle_emb_ndim": 512,
        "puzzle_emb_len": 16,
        "halt_exploration_prob": 0.1,
        "halt_max_steps": 16,
        "no_ACT_continue": True,
        "batch_size": args.batch_size,
        "vocab_size": metadata.vocab_size,
        "seq_len": metadata.seq_len,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
        "causal": False,
    }
    
    # Create base model
    base_model = TinyRecursiveReasoningModel_ACTV1(base_arch_cfg)
    
    # Load checkpoint
    base_model = load_base_trm_checkpoint(
        checkpoint_path=args.base_checkpoint,
        model=base_model,
        map_location=args.device,
        strict=False,
    )
    
    # Move to device
    base_model = base_model.to(args.device)
    
    # Freeze base weights, only LoRA will be trainable
    for name, param in base_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    print(f"Base model loaded. Trainable LoRA params: {sum(p.numel() for p in base_model.parameters() if p.requires_grad)}")
    
    # Train LoRA for each architecture combination
    total_archs = len(args.h_cycle_choices) * len(args.l_cycle_choices)
    print(f"\nTraining LoRAs for {total_archs} architecture combinations...")
    
    for h_cycles in args.h_cycle_choices:
        for l_cycles in args.l_cycle_choices:
            # Skip minimal architecture (already trained)
            if h_cycles == args.minimal_h_cycles and l_cycles == args.minimal_l_cycles:
                print(f"\nSkipping minimal architecture H={h_cycles} L={l_cycles} (already in base model)")
                continue
            
            print(f"\n{'='*60}")
            print(f"Training LoRA for H_cycles={h_cycles}, L_cycles={l_cycles}")
            print(f"{'='*60}")
            
            # Train LoRA
            lora_state_dict = train_lora_for_architecture(
                base_model=base_model,
                h_cycles=h_cycles,
                l_cycles=l_cycles,
                dataset=dataset,
                num_steps=args.num_steps,
                lr=args.lr,
                device=args.device,
            )
            
            # Save LoRA
            filename = f"lora_H{h_cycles}_L{l_cycles}.pth"
            filepath = os.path.join(args.lora_dir, filename)
            save_lora_state_dict(lora_state_dict, filepath)
            print(f"Saved LoRA to {filepath}")
    
    print(f"\n{'='*60}")
    print("All LoRAs trained and saved!")
    print(f"LoRA directory: {args.lora_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

