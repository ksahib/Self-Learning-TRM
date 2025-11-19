"""
Quick functional test for MetaTRM and its integration helpers.
"""

import os
import sys

import torch
import yaml

MODEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_ROOT, ".."))
for path in (MODEL_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.recursive_reasoning.metaTRM import MetaTRM
from models.recursive_reasoning.helper import (
    meta_trm_to_base_trm_pipeline,
    infer_vocab_size_from_checkpoint,
)


def _build_meta_config(batch_size: int, seq_len: int, vocab_size: int) -> dict:
    return dict(
        batch_size=batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=vocab_size,
        H_cycles=2,
        L_cycles=2,
        L_layers=2,
        hidden_size=64,
        expansion=4.0,
        num_heads=4,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        aug_slots=6,
        choices_per_slot=2,
    )


def _build_base_config(batch_size: int, seq_len: int, vocab_size: int) -> dict:
    return dict(
        batch_size=batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=vocab_size,
        H_cycles=1,
        L_cycles=1,
        H_layers=1,
        L_layers=2,
        hidden_size=64,
        expansion=4.0,
        num_heads=4,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=1,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        no_ACT_continue=True,
    )


def main():
    batch_size = 2
    seq_len = 81
    vocab_size = 11  # Match checkpoint vocab_size

    meta = MetaTRM(_build_meta_config(batch_size, seq_len, vocab_size))
    batch = {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long),
    }

    carry = meta.initial_carry(batch)
    carry, outputs = meta(
        carry,
        batch,
        sample=True,
        temperature=0.7,
    )
    print("Augmentation logits:", outputs["aug_logits"].shape)
    print("Sampled grid tensor:", outputs["sampled_grid"].shape)
    print("Sampled patterns:", outputs["sampled_patterns"])

    base = MetaTRM.build_base_trm(_build_base_config(batch_size, seq_len, vocab_size))
    base_carry, base_outputs = MetaTRM.run_base_trm_step(base, batch)
    print("Base TRM logits:", base_outputs["logits"].shape)
    
    # Test full pipeline connection
    print("\n" + "="*50)
    print("Testing MetaTRM → Base TRM pipeline connection:")
    print("="*50)
    
    # Add labels for the pipeline test
    batch_with_labels = {
        "inputs": batch["inputs"],
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": batch["puzzle_identifiers"],
    }
    
    patterns, log_probs, base_outputs_list = meta_trm_to_base_trm_pipeline(
        meta_model=meta,
        base_model=base,
        batch=batch_with_labels,
        sample=True,
        temperature=0.7,
        num_finetune_steps=0,  # Just evaluate, no fine-tuning for test
        grid_height=9,
        grid_width=9,
    )
    
    print(f"Number of patterns: {len(patterns)}")
    print(f"Patterns: {patterns}")
    print(f"Log probabilities shape: {log_probs.shape}")
    print(f"Number of base TRM outputs: {len(base_outputs_list)}")
    for i, output in enumerate(base_outputs_list):
        print(f"  Base TRM output {i+1} - logits shape: {output['logits'].shape}")
        print(f"  Base TRM output {i+1} - q_halt_logits shape: {output['q_halt_logits'].shape}")
    
    print("\n✓ Pipeline connection test passed!")
    
    # Test checkpoint loading
    print("\n" + "="*50)
    print("Testing checkpoint loading:")
    print("="*50)
    
    # Try multiple possible locations for checkpoint
    possible_checkpoint_paths = [
        os.path.join(REPO_ROOT, "TinyRecursiveModels", "step_21700"),  # In TinyRecursiveModels folder
        os.path.join(REPO_ROOT, "tryinghugs", "step_21700"),  # In tryinghugs folder
        os.path.join(REPO_ROOT, "step_21700"),  # Relative to REPO_ROOT
        os.path.join(MODEL_ROOT, "..", "step_21700"),  # Relative to MODEL_ROOT
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "step_21700"),  # Relative to test file
        "step_21700",  # Current directory
        os.path.join(os.getcwd(), "step_21700"),  # Absolute from CWD
    ]
    
    possible_config_paths = [
        os.path.join(REPO_ROOT, "TinyRecursiveModels", "all_config.yaml"),  # In TinyRecursiveModels folder
        os.path.join(REPO_ROOT, "tryinghugs", "all_config.yaml"),  # In tryinghugs folder
        os.path.join(REPO_ROOT, "all_config.yaml"),
        os.path.join(MODEL_ROOT, "..", "all_config.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "all_config.yaml"),
        "all_config.yaml",
        os.path.join(os.getcwd(), "all_config.yaml"),
    ]
    
    checkpoint_path = None
    config_path = None
    
    for path in possible_checkpoint_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            checkpoint_path = abs_path
            print(f"Found checkpoint at {checkpoint_path}")
            break
    
    for path in possible_config_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            config_path = abs_path
            break
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Infer vocab_size from checkpoint
        try:
            checkpoint_vocab_size = infer_vocab_size_from_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"Warning: Could not infer vocab_size from checkpoint: {e}")
            print(f"Using default vocab_size={vocab_size}")
            checkpoint_vocab_size = vocab_size
        
        # Try to read config from YAML, fall back to hardcoded values
        if config_path and os.path.exists(config_path):
            print(f"Reading config from {config_path}")
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                arch_config = yaml_config.get('arch', {})
                
                checkpoint_base_config = dict(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    puzzle_emb_ndim=arch_config.get('puzzle_emb_ndim', 512),
                    num_puzzle_identifiers=1,  # Not in YAML, use default
                    vocab_size=checkpoint_vocab_size,  # Use inferred vocab_size
                    H_cycles=arch_config.get('H_cycles', 3),
                    L_cycles=arch_config.get('L_cycles', 6),
                    H_layers=arch_config.get('H_layers', 0),
                    L_layers=arch_config.get('L_layers', 2),
                    hidden_size=arch_config.get('hidden_size', 512),
                    expansion=float(arch_config.get('expansion', 4)),
                    num_heads=arch_config.get('num_heads', 8),
                    pos_encodings=arch_config.get('pos_encodings', 'rope'),
                    rms_norm_eps=1e-5,  # Default
                    rope_theta=10000.0,  # Default
                    halt_max_steps=arch_config.get('halt_max_steps', 16),
                    halt_exploration_prob=arch_config.get('halt_exploration_prob', 0.1),
                    forward_dtype=arch_config.get('forward_dtype', 'bfloat16'),
                    mlp_t=arch_config.get('mlp_t', False),
                    puzzle_emb_len=arch_config.get('puzzle_emb_len', 16),
                    no_ACT_continue=arch_config.get('no_ACT_continue', True),
                )
                print("✓ Config loaded from YAML")
            except Exception as e:
                print(f"Warning: Failed to read YAML config: {e}")
                print("Using hardcoded config values")
                checkpoint_base_config = dict(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    puzzle_emb_ndim=512,
                    num_puzzle_identifiers=1,
                    vocab_size=checkpoint_vocab_size,  # Use inferred vocab_size
                    H_cycles=3,
                    L_cycles=6,
                    H_layers=0,
                    L_layers=2,
                    hidden_size=512,
                    expansion=4.0,
                    num_heads=8,
                    pos_encodings="rope",
                    rms_norm_eps=1e-5,
                    rope_theta=10000.0,
                    halt_max_steps=16,
                    halt_exploration_prob=0.1,
                    forward_dtype="bfloat16",
                    mlp_t=False,
                    puzzle_emb_len=16,
                    no_ACT_continue=True,
                )
        else:
            if config_path:
                print(f"Config file not found at {config_path}, using hardcoded values")
            else:
                print("Config file not found, using hardcoded values")
            checkpoint_base_config = dict(
                batch_size=batch_size,
                seq_len=seq_len,
                puzzle_emb_ndim=512,
                num_puzzle_identifiers=1,
                vocab_size=checkpoint_vocab_size,  # Use inferred vocab_size
                H_cycles=3,
                L_cycles=6,
                H_layers=0,
                L_layers=2,
                hidden_size=512,
                expansion=4.0,
                num_heads=8,
                pos_encodings="rope",
                rms_norm_eps=1e-5,
                rope_theta=10000.0,
                halt_max_steps=16,
                halt_exploration_prob=0.1,
                forward_dtype="bfloat16",
                mlp_t=False,
                puzzle_emb_len=16,
                no_ACT_continue=True,
            )
        
        try:
            # Load checkpoint
            loaded_base = MetaTRM.load_base_trm_from_checkpoint(
                checkpoint_path=checkpoint_path,
                config_dict=checkpoint_base_config,
                map_location="cpu",
                strict=False,
            )
            
            # Test forward pass with loaded model (use checkpoint vocab_size)
            # Use batch_size=2 to match the rest of the test and avoid shape mismatches
            test_batch_size = 2
            test_batch = {
                "inputs": torch.randint(0, checkpoint_vocab_size, (test_batch_size, seq_len)),
                "puzzle_identifiers": torch.zeros(test_batch_size, dtype=torch.long),
            }
            loaded_carry, loaded_outputs = MetaTRM.run_base_trm_step(loaded_base, test_batch)
            print(f"Loaded model forward pass - logits shape: {loaded_outputs['logits'].shape}")
            print("✓ Checkpoint loading test passed!")
        except Exception as e:
            print(f"✗ Checkpoint loading failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Checkpoint not found. Tried the following paths:")
        for path in possible_checkpoint_paths:
            abs_path = os.path.abspath(path)
            print(f"  - {abs_path} {'✓' if os.path.exists(abs_path) else '✗'}")
        print("Skipping checkpoint loading test.")
    
    print("\nFinished test run.")


if __name__ == "__main__":
    main()


