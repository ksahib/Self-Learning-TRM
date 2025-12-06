"""
Build vector database by encoding all training puzzles using TRM model.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
from omegaconf import DictConfig

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.vector_db import PuzzleVectorDB
from utils.functions import load_model_class
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def encode_puzzle_batch(
    model: TinyRecursiveReasoningModel_ACTV1,
    batch: dict,
    device: str = "cuda",
) -> np.ndarray:
    """
    Encode a batch of puzzles to get embeddings.
    Uses the input embeddings from the model.
    
    Args:
        model: TRM model
        batch: Batch dict with 'inputs' and 'puzzle_identifiers'
        device: Device to run on
    
    Returns:
        Embeddings [batch_size, embedding_dim]
    """
    model.eval()
    with torch.no_grad():
        # Move batch to device
        inputs = batch["inputs"].to(device)
        puzzle_identifiers = batch["puzzle_identifiers"].to(device)
        
        # Get input embeddings
        # Access the inner model's _input_embeddings method
        input_embeddings = model.inner._input_embeddings(inputs, puzzle_identifiers)
        
        # Use mean pooling over sequence dimension to get fixed-size embedding
        # Shape: [batch_size, seq_len + puzzle_emb_len, hidden_size]
        # Pool to: [batch_size, hidden_size]
        embeddings = input_embeddings.mean(dim=1)  # Mean pool over sequence
        
        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy()
        
        return embeddings_np


def build_vector_database(
    data_paths: list,
    model_checkpoint: str,
    output_path: str,
    arch_config: dict,
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 0,
):
    """
    Build vector database by encoding all training puzzles.
    
    Args:
        data_paths: List of dataset paths
        model_checkpoint: Path to TRM model checkpoint (or None for random init)
        output_path: Path to save vector database
        arch_config: Model architecture config dict
        device: Device to use
        batch_size: Batch size for encoding
        seed: Random seed
    """
    print("Building vector database...")
    print(f"Data paths: {data_paths}")
    print(f"Output path: {output_path}")
    
    # Load dataset metadata
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
    
    print(f"Dataset metadata: {metadata}")
    
    # Create model
    model_cfg = dict(
        **arch_config,
        batch_size=batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
    model = model.to(device)
    
    # Load checkpoint if provided
    if model_checkpoint and os.path.exists(model_checkpoint):
        print(f"Loading checkpoint from {model_checkpoint}")
        checkpoint = torch.load(model_checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Remove _orig_mod prefixes if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod.model."):
                new_key = key[len("_orig_mod.model."):]
            elif key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod."):]
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded")
    else:
        print("No checkpoint provided, using randomly initialized model")
    
    # Collect all puzzles
    print("Encoding puzzles...")
    all_embeddings = []
    all_puzzle_indices = []
    all_puzzle_identifiers = []
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=4,
    )
    
    # Encode all puzzles
    puzzle_index_counter = 0
    for set_name, batch, global_batch_size in tqdm(dataloader, desc="Encoding"):
        # Encode batch
        embeddings = encode_puzzle_batch(model, batch, device=device)
        
        # Get puzzle identifiers
        puzzle_identifiers = batch["puzzle_identifiers"].cpu().numpy()
        
        # For each example in batch, track its puzzle
        batch_size_actual = batch["inputs"].shape[0]
        
        # Get unique puzzle identifiers in this batch
        unique_puzzle_ids = np.unique(puzzle_identifiers)
        
        # For each unique puzzle, get one representative embedding
        # (use first occurrence of each puzzle)
        for puzzle_id in unique_puzzle_ids:
            mask = puzzle_identifiers == puzzle_id
            if np.any(mask):
                # Use first occurrence
                idx = np.where(mask)[0][0]
                all_embeddings.append(embeddings[idx])
                all_puzzle_indices.append(puzzle_index_counter)
                all_puzzle_identifiers.append(int(puzzle_id))
                puzzle_index_counter += 1
    
    # Convert to numpy arrays
    all_embeddings = np.array(all_embeddings)
    all_puzzle_indices = np.array(all_puzzle_indices)
    all_puzzle_identifiers = np.array(all_puzzle_identifiers)
    
    print(f"Encoded {len(all_embeddings)} unique puzzles")
    print(f"Embedding shape: {all_embeddings.shape}")
    
    # Create vector database
    vector_db = PuzzleVectorDB(
        embeddings=all_embeddings,
        puzzle_indices=all_puzzle_indices,
        puzzle_identifiers=all_puzzle_identifiers,
        metadata={
            "seq_len": metadata.seq_len,
            "vocab_size": metadata.vocab_size,
            "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
            "data_paths": data_paths,
        },
    )
    
    # Save
    vector_db.save(output_path)
    print(f"Vector database saved to {output_path}")
    
    return vector_db


def main():
    parser = argparse.ArgumentParser(description="Build vector database for puzzle similarity search")
    parser.add_argument("--data_paths", type=str, nargs="+", required=True, help="Dataset paths")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for vector database")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to model checkpoint (optional)")
    parser.add_argument("--arch_config", type=str, default=None, help="Path to arch config YAML")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    # Load arch config if provided
    if args.arch_config:
        with open(args.arch_config, "r") as f:
            arch_config = yaml.safe_load(f)
    else:
        # Default config (will need to be provided)
        raise ValueError("--arch_config is required")
    
    build_vector_database(
        data_paths=args.data_paths,
        model_checkpoint=args.model_checkpoint,
        output_path=args.output_path,
        arch_config=arch_config,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

