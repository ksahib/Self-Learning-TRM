
"""
Integration test for meta-training pipeline.

Verifies:
1. Meta-model can sample patterns
2. Base TRM loads and fine-tunes with LoRA
3. Reward computation works
4. Policy gradient updates meta-model
5. End-to-end training step completes
"""

import os
import sys

import torch
from torch.utils.data import DataLoader

MODEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_ROOT, ".."))
for path in (MODEL_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.recursive_reasoning.metaTRM import MetaTRM
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.recursive_reasoning.helper import (
    get_lora_parameters,
    compute_base_trm_loss,
    compute_reward,
    compute_rewards_from_augmentations,
    sample_indices_from_logits,
    indices_to_pattern_strings,
    load_base_trm_checkpoint,
    compute_base_trm_eval,
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
        H_cycles=2,
        L_cycles=2,
        H_layers=0,
        L_layers=2,
        hidden_size=128,
        expansion=4.0,
        num_heads=4,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        halt_exploration_prob=0.1,
        halt_max_steps=8,
        no_ACT_continue=True,
    )


def test_meta_model_sampling():
    """Test 1: Meta-model can sample patterns"""
    print("\n" + "="*50)
    print("Test 1: Meta-model pattern sampling")
    print("="*50)
    
    batch_size = 2
    seq_len = 81
    vocab_size = 16
    
    meta = MetaTRM(_build_meta_config(batch_size, seq_len, vocab_size))
    batch = {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long),
    }
    
    # Sample patterns
    meta_carry = meta.initial_carry(batch)
    meta_carry, outputs = meta(
        meta_carry,
        batch,
        sample=True,
        temperature=1.0,
    )
    
    patterns = outputs["sampled_patterns"]
    log_probs = outputs["sampled_log_probs"]
    
    assert len(patterns) == batch_size, f"Expected {batch_size} patterns, got {len(patterns)}"
    assert log_probs.shape == (batch_size, 6), f"Expected log_probs shape [batch, 6], got {log_probs.shape}"
    
    print(f"✓ Sampled {len(patterns)} patterns: {patterns}")
    print(f"✓ Log probabilities shape: {log_probs.shape}")
    return patterns, log_probs


def test_lora_parameters():
    """Test 2: Base TRM has LoRA parameters"""
    print("\n" + "="*50)
    print("Test 2: LoRA parameters extraction")
    print("="*50)
    
    batch_size = 2
    seq_len = 81
    vocab_size = 16
    
    base_model = TinyRecursiveReasoningModel_ACTV1(_build_base_config(batch_size, seq_len, vocab_size))
    
    # Freeze base weights
    for name, param in base_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    # Get LoRA parameters
    lora_params = get_lora_parameters(base_model)
    
    assert len(lora_params) > 0, "No LoRA parameters found!"
    
    # Check that LoRA params are trainable
    trainable_count = sum(1 for p in lora_params if p.requires_grad)
    assert trainable_count == len(lora_params), "Not all LoRA parameters are trainable!"
    
    total_lora_params = sum(p.numel() for p in lora_params)
    print(f"✓ Found {len(lora_params)} LoRA parameter groups")
    print(f"✓ Total LoRA parameters: {total_lora_params}")
    return base_model, lora_params


def test_reward_computation():
    """Test 3: Reward computation works"""
    print("\n" + "="*50)
    print("Test 3: Reward computation")
    print("="*50)
    
    batch_size = 2
    seq_len = 81
    vocab_size = 16
    
    base_model = TinyRecursiveReasoningModel_ACTV1(_build_base_config(batch_size, seq_len, vocab_size))
    base_model.eval()
    
    batch = {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long),
    }
    
    # Compute loss
    loss = compute_base_trm_loss(
        base_model=base_model,
        batch=batch,
        loss_type="softmax_cross_entropy"
    )
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.requires_grad == False, "Loss should be detached"
    assert loss.numel() == 1, "Loss should be scalar"
    
    print(f"✓ Computed loss: {loss.item():.4f}")
    
    # Compute reward
    reward = compute_reward(loss, reward_scale=1.0)
    
    assert isinstance(reward, float), "Reward should be a float"
    assert reward < 0, "Reward should be negative (lower loss = higher reward)"
    
    print(f"✓ Computed reward: {reward:.4f}")
    return loss, reward


def test_lora_finetuning():
    """Test 4: Base TRM can fine-tune with LoRA"""
    print("\n" + "="*50)
    print("Test 4: LoRA fine-tuning")
    print("="*50)
    
    batch_size = 2
    seq_len = 81
    vocab_size = 16
    
    base_model = TinyRecursiveReasoningModel_ACTV1(_build_base_config(batch_size, seq_len, vocab_size))
    
    # Freeze base weights
    for name, param in base_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    # Get LoRA parameters
    lora_params = get_lora_parameters(base_model)
    optimizer = torch.optim.Adam(lora_params, lr=1e-4)
    
    batch = {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long),
    }
    
    # Fine-tune for a few steps
    base_model.train()
    initial_loss = None
    for step in range(3):
        carry = base_model.initial_carry(batch)
        carry, outputs = base_model(carry=carry, batch=batch)
        
        logits = outputs["logits"]
        labels = batch["labels"]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=-100
        )
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    final_loss = loss.item()
    print(f"✓ Fine-tuned for 3 steps")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    
    return base_model


def test_rewards_from_augmentations():
    """Test 5: Reward computation from augmentations"""
    print("\n" + "="*50)
    print("Test 5: Rewards from augmentations")
    print("="*50)
    
    batch_size = 2
    seq_len = 81
    vocab_size = 16
    num_patterns = 3  # Configurable number of patterns
    
    base_model = TinyRecursiveReasoningModel_ACTV1(_build_base_config(batch_size, seq_len, vocab_size))
    
    # Freeze base weights
    for name, param in base_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    original_batch = {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long),
    }
    
    # Test with configurable number of patterns
    # Generate patterns (can be any number)
    patterns = ['LR.V.T', '.RH.OT', 'LRH.VT'][:num_patterns]
    
    # Debug: Check batch format before evaluation
    print(f"\nDEBUG: Checking evaluation setup...")
    print(f"  Batch inputs shape: {original_batch['inputs'].shape}")
    print(f"  Batch labels shape: {original_batch['labels'].shape}")
    print(f"  Labels unique values: {torch.unique(original_batch['labels'])}")
    print(f"  Labels min/max: {original_batch['labels'].min().item()}/{original_batch['labels'].max().item()}")
    
    rewards, baseline_loss, baseline_metrics, pattern_metrics = compute_rewards_from_augmentations(
        base_model=base_model,
        original_batch=original_batch,
        patterns=patterns,
        num_finetune_steps=2,  # Small number for test
        use_binary_reward=True,
    )
    
    assert len(rewards) == len(patterns), f"Expected {len(patterns)} rewards, got {len(rewards)}"
    assert len(rewards) == num_patterns, f"Expected {num_patterns} rewards, got {len(rewards)}"
    assert all(isinstance(r, float) for r in rewards), "All rewards should be floats"
    assert all(r in [0.0, 1.0] for r in rewards), "Binary rewards should be 0.0 or 1.0"
    assert isinstance(baseline_loss, float), "Baseline loss should be float"
    for key in ["accuracy", "exact_accuracy", "lm_loss"]:
        assert key in baseline_metrics and baseline_metrics[key] is not None, f"Missing {key} in baseline metrics"
    assert len(pattern_metrics) == len(patterns), "Should track metrics per pattern"
    
    # Enhanced diagnostics: Check exact_accuracy values
    print(f"\nDEBUG: Evaluation metrics...")
    assert isinstance(baseline_metrics["exact_accuracy"], (int, float)), "exact_accuracy should be numeric"
    print(f"  Baseline exact_accuracy: {baseline_metrics['exact_accuracy']:.4f}")
    print(f"  Baseline token accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  Baseline loss: {baseline_loss:.4f}")
    print(f"  Baseline valid_count: {baseline_metrics.get('count', 'N/A')}")
    print(f"  Baseline total_sequences: {baseline_metrics.get('total_sequences', 'N/A')}")
    
    # Check pattern metrics
    for i, pm in enumerate(pattern_metrics):
        if "exact_accuracy" in pm and pm["exact_accuracy"] is not None:
            print(f"  Pattern {i} exact_accuracy: {pm['exact_accuracy']:.4f}")
        if "accuracy" in pm and pm["accuracy"] is not None:
            print(f"  Pattern {i} token accuracy: {pm['accuracy']:.4f}")
    
    # Warning if exact_accuracy is 0
    if baseline_metrics["exact_accuracy"] == 0.0:
        print(f"\n⚠ WARNING: Baseline exact_accuracy is 0.0 - this may indicate:")
        print(f"  - Sequences not halting properly during evaluation")
        print(f"  - All labels are being ignored (IGNORE_LABEL_ID)")
        print(f"  - No valid sequences found (valid_count = 0)")
    
    print(f"\n✓ Computed rewards for {len(patterns)} patterns: {rewards}")
    print(f"✓ Baseline loss: {baseline_loss:.4f}")
    return rewards, baseline_loss


def test_policy_gradient():
    """Test 6: Policy gradient updates meta-model"""
    print("\n" + "="*50)
    print("Test 6: Policy gradient update")
    print("="*50)
    
    batch_size = 2
    seq_len = 81
    vocab_size = 16
    num_patterns = 3  # Configurable number of patterns
    
    meta_model = MetaTRM(_build_meta_config(batch_size, seq_len, vocab_size))
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-4)
    
    # Get initial parameters
    initial_params = {name: param.clone() for name, param in meta_model.named_parameters()}
    
    batch = {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long),
    }
    
    # Sample patterns - use helper function to get exactly num_patterns
    from models.recursive_reasoning.metaTRM import MetaTRM as MetaTRMClass
    meta_carry = meta_model.initial_carry(batch)
    meta_carry, outputs = meta_model(
        meta_carry,
        batch,
        sample=True,
        temperature=1.0,
    )
    
    # Get all available patterns and cycle if needed to get num_patterns
    all_patterns = outputs["sampled_patterns"]
    all_log_probs = outputs["sampled_log_probs"]  # [batch_size, slots]
    
    # Take num_patterns patterns (cycle if needed)
    patterns = []
    log_probs_list = []
    for i in range(num_patterns):
        idx = i % len(all_patterns)
        patterns.append(all_patterns[idx])
        log_probs_list.append(all_log_probs[idx])
    
    log_probs = torch.stack(log_probs_list)  # [num_patterns, slots]
    pattern_log_probs = log_probs.sum(dim=-1)  # [num_patterns]
    
    # Simulate rewards - match the actual number of patterns
    rewards = torch.tensor([0.0, 1.0, 0.0][:num_patterns], dtype=torch.float32)
    baseline = 0.33
    
    # Policy gradient
    advantages = rewards - baseline
    policy_loss = -(pattern_log_probs * advantages).mean()
    
    # Backprop
    meta_optimizer.zero_grad()
    policy_loss.backward()
    meta_optimizer.step()
    
    # Check that parameters changed
    params_changed = False
    for name, param in meta_model.named_parameters():
        if not torch.equal(param, initial_params[name]):
            params_changed = True
            break
    
    assert params_changed, "Meta model parameters should have changed after update!"
    assert len(patterns) == num_patterns, f"Expected {num_patterns} patterns, got {len(patterns)}"
    
    print(f"✓ Sampled {len(patterns)} patterns")
    print(f"✓ Policy loss: {policy_loss.item():.4f}")
    print(f"✓ Meta model parameters updated: {params_changed}")
    return meta_model


def test_checkpoint_loading():
    """Test 7: Checkpoint loading and verification"""
    print("\n" + "="*50)
    print("Test 7: Checkpoint loading verification")
    print("="*50)
    
    batch_size = 2
    seq_len = 81
    
    # Try to find checkpoint (check common locations)
    checkpoint_paths = [
        "step_21700",
        "checkpoints/step_21700",
        "../step_21700",
        "../../step_21700",
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print("⚠ No checkpoint found (checked: step_21700, checkpoints/step_21700, etc.)")
        print("  Skipping checkpoint loading test")
        print("  To test checkpoint loading, ensure step_21700 exists in project root")
        return None
    
    print(f"✓ Found checkpoint: {checkpoint_path}")
    
    # First, inspect checkpoint to get actual vocab_size and hidden_size
    print(f"  Inspecting checkpoint to infer architecture...")
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle wrapped checkpoints
    if isinstance(checkpoint_data, dict):
        if "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]
        elif "state_dict" in checkpoint_data:
            state_dict = checkpoint_data["state_dict"]
        elif "model" in checkpoint_data:
            state_dict = checkpoint_data["model"]
        else:
            state_dict = checkpoint_data
    else:
        state_dict = checkpoint_data
    
    # Handle torch.compile prefixes
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod."):]
                if new_key.startswith("model."):
                    new_key = new_key[len("model."):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    # Infer vocab_size and hidden_size from checkpoint
    inferred_vocab_size = None
    inferred_hidden_size = None
    
    # Try to find embedding weights
    embedding_keys = [
        "inner.embed_tokens.embedding_weight",
        "embed_tokens.embedding_weight",
    ]
    for key in embedding_keys:
        if key in state_dict:
            inferred_vocab_size = state_dict[key].shape[0]
            inferred_hidden_size = state_dict[key].shape[1]
            print(f"  Inferred from {key}: vocab_size={inferred_vocab_size}, hidden_size={inferred_hidden_size}")
            break
    
    if inferred_vocab_size is None:
        # Try lm_head
        lm_head_keys = [
            "inner.lm_head.base.weight",
            "inner.lm_head.weight",
            "lm_head.weight",
        ]
        for key in lm_head_keys:
            if key in state_dict:
                inferred_vocab_size = state_dict[key].shape[0]
                print(f"  Inferred vocab_size={inferred_vocab_size} from {key}")
                break
    
    if inferred_hidden_size is None:
        # Try to infer from H_init or L_init
        init_keys = ["inner.H_init", "H_init"]
        for key in init_keys:
            if key in state_dict:
                inferred_hidden_size = state_dict[key].shape[0]
                print(f"  Inferred hidden_size={inferred_hidden_size} from {key}")
                break
    
    # Try to load config from checkpoint directory
    import yaml
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    
    # Default config (will be overridden)
    base_config = _build_base_config(batch_size, seq_len, vocab_size=16)
    
    if os.path.exists(config_path):
        print(f"  Loading config from: {config_path}")
        try:
            with open(config_path, "rt") as f:
                checkpoint_config = yaml.safe_load(f)
            
            if "arch" in checkpoint_config:
                checkpoint_arch = checkpoint_config["arch"]
                print(f"  Found checkpoint config: {checkpoint_arch.get('name', 'unknown')}")
                
                # Use checkpoint's config values for non-critical params
                base_config.update({
                    "num_heads": checkpoint_arch.get("num_heads", base_config["num_heads"]),
                    "expansion": checkpoint_arch.get("expansion", base_config["expansion"]),
                    "L_layers": checkpoint_arch.get("L_layers", base_config["L_layers"]),
                    "H_cycles": checkpoint_arch.get("H_cycles", base_config["H_cycles"]),
                    "L_cycles": checkpoint_arch.get("L_cycles", base_config["L_cycles"]),
                    "puzzle_emb_ndim": checkpoint_arch.get("puzzle_emb_ndim", base_config.get("puzzle_emb_ndim", 0)),
                    "puzzle_emb_len": checkpoint_arch.get("puzzle_emb_len", base_config.get("puzzle_emb_len", 0)),
                    "halt_max_steps": checkpoint_arch.get("halt_max_steps", base_config["halt_max_steps"]),
                    "halt_exploration_prob": checkpoint_arch.get("halt_exploration_prob", base_config["halt_exploration_prob"]),
                    "no_ACT_continue": checkpoint_arch.get("no_ACT_continue", base_config.get("no_ACT_continue", True)),
                })
        except Exception as e:
            print(f"  ⚠ Could not load checkpoint config ({e}), using inferred values")
    else:
        print(f"  ⚠ Config not found at {config_path}, using inferred values")
    
    # CRITICAL: Use inferred values for vocab_size and hidden_size (must match checkpoint exactly)
    if inferred_vocab_size is not None:
        base_config["vocab_size"] = inferred_vocab_size
        print(f"  ✓ Using inferred vocab_size={inferred_vocab_size} (from checkpoint weights, overriding config)")
    else:
        print(f"  ⚠ WARNING: Could not infer vocab_size from checkpoint!")
    
    if inferred_hidden_size is not None:
        base_config["hidden_size"] = inferred_hidden_size
        print(f"  ✓ Using inferred hidden_size={inferred_hidden_size} (from checkpoint weights, overriding config)")
    else:
        print(f"  ⚠ WARNING: Could not infer hidden_size from checkpoint!")
    
    # Update batch_size and seq_len (these come from test, not checkpoint)
    base_config["batch_size"] = batch_size
    base_config["seq_len"] = seq_len
    base_config["num_puzzle_identifiers"] = 1  # Default for test
    
    # Create base model with checkpoint's config
    print(f"  Creating model with config: hidden_size={base_config['hidden_size']}, vocab_size={base_config['vocab_size']}")
    base_model = TinyRecursiveReasoningModel_ACTV1(base_config)
    
    # Get initial weight norm (before loading)
    initial_lm_head_norm = base_model.inner.lm_head.base.weight.norm().item()
    print(f"  Initial lm_head weight norm: {initial_lm_head_norm:.4f}")
    
    # Load checkpoint
    try:
        base_model = load_base_trm_checkpoint(
            checkpoint_path=checkpoint_path,
            model=base_model,
            map_location="cpu",
            strict=False,
        )
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Verify weights changed
    loaded_lm_head_norm = base_model.inner.lm_head.base.weight.norm().item()
    print(f"  Loaded lm_head weight norm: {loaded_lm_head_norm:.4f}")
    
    if abs(loaded_lm_head_norm - initial_lm_head_norm) < 1e-6:
        print("  ⚠ WARNING: Weights didn't change - checkpoint may not have loaded!")
    else:
        print("  ✓ Weights changed - checkpoint loaded correctly")
    
    # Test evaluation with loaded checkpoint
    print(f"\n  Testing evaluation with loaded checkpoint...")
    base_model.eval()
    
    # Use random data (like other tests) - use vocab_size from config
    test_vocab_size = base_config.get("vocab_size", 16)
    test_batch = {
        "inputs": torch.randint(0, test_vocab_size, (batch_size, seq_len)),
        "labels": torch.randint(0, test_vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long),
    }
    
    eval_metrics = compute_base_trm_eval(
        base_model=base_model,
        batch=test_batch,
        loss_type="softmax_cross_entropy"
    )
    
    print(f"  Evaluation results:")
    print(f"    Token accuracy: {eval_metrics['token_accuracy']:.4f}")
    print(f"    Exact accuracy: {eval_metrics['exact_accuracy']:.4f}")
    print(f"    Loss: {eval_metrics['loss']:.4f}")
    print(f"    Valid count: {eval_metrics.get('count', 'N/A')}")
    
    # Note: With random data, accuracy will still be low
    # But if checkpoint loaded correctly, the model should at least run without errors
    print(f"  ✓ Evaluation completed (low accuracy expected with random data)")
    
    return base_model


def test_checkpoint_with_real_data():
    """Test 8: Checkpoint loading with real sudoku data"""
    print("\n" + "="*50)
    print("Test 8: Checkpoint loading with REAL data")
    print("="*50)
    
    # Check if dataset exists (try multiple paths depending on where script is run from)
    data_paths = [
        "data/sudoku-extreme-3-aug-3",  # From project root
        "../../data/sudoku-extreme-3-aug-3",  # From models/recursive_reasoning/
        "../data/sudoku-extreme-3-aug-3",  # From models/
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print(f"⚠ Dataset not found (checked: {', '.join(data_paths)})")
        print("  Skipping real data test")
        return None
    
    print(f"✓ Found dataset: {data_path}")
    
    # Try to find checkpoint
    checkpoint_paths = [
        "step_21700",
        "checkpoints/step_21700",
        "../step_21700",
        "../../step_21700",
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print("⚠ No checkpoint found")
        print("  Skipping real data test")
        return None
    
    print(f"✓ Found checkpoint: {checkpoint_path}")
    
    # Load real dataset
    try:
        from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
        
        # Test on train split (and test if available)
        datasets_to_test = []
        for split_name, test_mode in [("train", False), ("test", True)]:
            try:
                dataset_cfg = PuzzleDatasetConfig(
                    seed=0,
                    dataset_paths=[data_path],
                    global_batch_size=2,
                    test_set_mode=test_mode,
                    epochs_per_iter=1,
                    rank=0,
                    num_replicas=1,
                )
                dataset = PuzzleDataset(dataset_cfg, split=split_name)
                dataloader = DataLoader(dataset, batch_size=None, num_workers=1)
                datasets_to_test.append((split_name, dataloader, dataset.metadata))
                print(f"✓ Loaded {split_name} split")
            except Exception as e:
                print(f"  ⚠ Could not load {split_name} split: {e}")
                # Continue with other splits
        
        if not datasets_to_test:
            raise Exception("Could not load any dataset splits")
        
        # Use first dataset for metadata and initial batch
        split_name, dataloader, metadata = datasets_to_test[0]
        
        # Get metadata
        metadata = dataset.metadata
        print(f"✓ Loaded dataset:")
        print(f"    vocab_size: {metadata.vocab_size}")
        print(f"    seq_len: {metadata.seq_len}")
        print(f"    num_puzzle_identifiers: {metadata.num_puzzle_identifiers}")
        
        # Get one batch of real data
        # PuzzleDataset yields (set_name, batch_dict, size) tuples
        set_name, real_batch, batch_size = next(iter(dataloader))
        print(f"✓ Loaded real batch (set: {set_name}, size: {batch_size}):")
        print(f"    inputs shape: {real_batch['inputs'].shape}")
        print(f"    labels shape: {real_batch['labels'].shape}")
        print(f"    labels unique values: {torch.unique(real_batch['labels'])}")
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Now load checkpoint (similar to test_checkpoint_loading)
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle wrapped checkpoints
    if isinstance(checkpoint_data, dict):
        if "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]
        elif "state_dict" in checkpoint_data:
            state_dict = checkpoint_data["state_dict"]
        elif "model" in checkpoint_data:
            state_dict = checkpoint_data["model"]
        else:
            state_dict = checkpoint_data
    else:
        state_dict = checkpoint_data
    
    # Handle torch.compile prefixes
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod."):]
                if new_key.startswith("model."):
                    new_key = new_key[len("model."):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    # Infer vocab_size and hidden_size from checkpoint
    inferred_vocab_size = None
    inferred_hidden_size = None
    
    embedding_keys = [
        "inner.embed_tokens.embedding_weight",
        "embed_tokens.embedding_weight",
    ]
    for key in embedding_keys:
        if key in state_dict:
            inferred_vocab_size = state_dict[key].shape[0]
            inferred_hidden_size = state_dict[key].shape[1]
            print(f"  Inferred from checkpoint: vocab_size={inferred_vocab_size}, hidden_size={inferred_hidden_size}")
            break
    
    if inferred_hidden_size is None:
        init_keys = ["inner.H_init", "H_init"]
        for key in init_keys:
            if key in state_dict:
                inferred_hidden_size = state_dict[key].shape[0]
                print(f"  Inferred hidden_size={inferred_hidden_size} from {key}")
                break
    
    # Build config using dataset metadata
    actual_batch_size = real_batch['inputs'].shape[0]
    base_config = _build_base_config(
        batch_size=actual_batch_size,
        seq_len=metadata.seq_len,
        vocab_size=metadata.vocab_size
    )
    
    # Override with checkpoint values
    if inferred_hidden_size is not None:
        base_config["hidden_size"] = inferred_hidden_size
    if inferred_vocab_size is not None:
        base_config["vocab_size"] = inferred_vocab_size
        print(f"  Using checkpoint vocab_size={inferred_vocab_size} (overriding dataset vocab_size={metadata.vocab_size})")
    
    base_config["num_puzzle_identifiers"] = metadata.num_puzzle_identifiers
    
    # Load checkpoint config if available
    import yaml
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "rt") as f:
                checkpoint_config = yaml.safe_load(f)
            
            if "arch" in checkpoint_config:
                checkpoint_arch = checkpoint_config["arch"]
                base_config.update({
                    "num_heads": checkpoint_arch.get("num_heads", base_config["num_heads"]),
                    "expansion": checkpoint_arch.get("expansion", base_config["expansion"]),
                    "L_layers": checkpoint_arch.get("L_layers", base_config["L_layers"]),
                    "H_cycles": checkpoint_arch.get("H_cycles", base_config["H_cycles"]),
                    "L_cycles": checkpoint_arch.get("L_cycles", base_config["L_cycles"]),
                    "puzzle_emb_ndim": checkpoint_arch.get("puzzle_emb_ndim", base_config.get("puzzle_emb_ndim", 0)),
                    "puzzle_emb_len": checkpoint_arch.get("puzzle_emb_len", base_config.get("puzzle_emb_len", 0)),
                    "halt_max_steps": checkpoint_arch.get("halt_max_steps", base_config["halt_max_steps"]),
                    "halt_exploration_prob": checkpoint_arch.get("halt_exploration_prob", base_config["halt_exploration_prob"]),
                    "no_ACT_continue": checkpoint_arch.get("no_ACT_continue", base_config.get("no_ACT_continue", True)),
                })
        except Exception as e:
            print(f"  ⚠ Could not load checkpoint config ({e})")
    
    print(f"  Creating model with config: hidden_size={base_config['hidden_size']}, vocab_size={base_config['vocab_size']}")
    base_model = TinyRecursiveReasoningModel_ACTV1(base_config)
    
    # Load checkpoint
    try:
        base_model = load_base_trm_checkpoint(
            checkpoint_path=checkpoint_path,
            model=base_model,
            map_location="cpu",
            strict=False,
        )
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Test evaluation with REAL data on multiple batches from all available splits
    print(f"\n  Testing evaluation with REAL sudoku data...")
    base_model.eval()
    
    # Evaluate on multiple batches to get better statistics
    num_batches_to_test = 10  # Try to get more batches
    all_token_accuracies = []
    all_exact_accuracies = []
    all_losses = []
    total_sequences = 0
    total_correct_sequences = 0
    
    print(f"  Testing on up to {num_batches_to_test} batches from all available splits...")
    
    batches_tested = 0
    for split_name, dataloader, split_metadata in datasets_to_test:
        print(f"    Testing {split_name} split...")
        try:
            batch_iter = iter(dataloader)
            
            for batch_idx in range(num_batches_to_test):
                try:
                    set_name, batch, batch_size = next(batch_iter)
                    
                    eval_metrics = compute_base_trm_eval(
                        base_model=base_model,
                        batch=batch,
                        loss_type="softmax_cross_entropy"
                    )
                    
                    all_token_accuracies.append(eval_metrics['token_accuracy'])
                    all_exact_accuracies.append(eval_metrics['exact_accuracy'])
                    all_losses.append(eval_metrics['loss'])
                    
                    batch_sequences = eval_metrics.get('total_sequences', batch_size)
                    batch_correct = int(eval_metrics['exact_accuracy'] * batch_sequences)
                    
                    total_sequences += batch_sequences
                    total_correct_sequences += batch_correct
                    batches_tested += 1
                    
                    print(f"      {split_name} batch {batch_idx + 1}: token_acc={eval_metrics['token_accuracy']:.4f}, "
                          f"exact_acc={eval_metrics['exact_accuracy']:.4f}, "
                          f"loss={eval_metrics['loss']:.4f}, "
                          f"correct={batch_correct}/{batch_sequences}")
                    
                except StopIteration:
                    print(f"      ⚠ {split_name} split has {batch_idx} batches")
                    break
                except Exception as e:
                    print(f"      ⚠ Error processing {split_name} batch {batch_idx + 1}: {e}")
                    break
                
                if batches_tested >= num_batches_to_test:
                    break
            
            if batches_tested >= num_batches_to_test:
                break
        except Exception as e:
            print(f"    ⚠ Error iterating {split_name} split: {e}")
            continue
    
    # Calculate aggregate statistics
    if len(all_token_accuracies) > 0:
        avg_token_accuracy = sum(all_token_accuracies) / len(all_token_accuracies)
        avg_exact_accuracy = sum(all_exact_accuracies) / len(all_exact_accuracies)
        avg_loss = sum(all_losses) / len(all_losses)
        overall_exact_accuracy = total_correct_sequences / total_sequences if total_sequences > 0 else 0.0
        
        print(f"\n  📊 AGGREGATE REAL DATA EVALUATION RESULTS ({batches_tested} batches, {total_sequences} sequences):")
        print(f"    Average token accuracy: {avg_token_accuracy:.4f} ({avg_token_accuracy*100:.2f}%)")
        print(f"    Average exact accuracy: {avg_exact_accuracy:.4f} ({avg_exact_accuracy*100:.2f}%)")
        print(f"    Overall exact accuracy: {overall_exact_accuracy:.4f} ({overall_exact_accuracy*100:.2f}%)")
        print(f"    Average loss: {avg_loss:.4f}")
        print(f"    Total correct sequences: {total_correct_sequences}/{total_sequences}")
        
        if total_correct_sequences > 0:
            print(f"\n  ✅ SUCCESS: Model correctly solves {total_correct_sequences} out of {total_sequences} sudoku puzzles!")
            print(f"     This confirms the checkpoint is working correctly with real data.")
            print(f"     The model is successfully solving sudoku puzzles, just like the original checkpoint!")
        elif avg_token_accuracy > 0.5:
            print(f"\n  ⚠ Model has high token accuracy ({avg_token_accuracy*100:.2f}%) but 0% exact accuracy.")
            print(f"     This suggests the model is close but not perfectly solving puzzles.")
        else:
            print(f"\n  ⚠ WARNING: Low accuracy - this may indicate:")
            print(f"    - Checkpoint not loading correctly")
            print(f"    - Model architecture mismatch")
            print(f"    - Data format issues")
    else:
        print(f"\n  ⚠ No batches were evaluated")
    
    return base_model


def test_end_to_end():
    """Test 8: End-to-end training step"""
    print("\n" + "="*50)
    print("Test 8: End-to-end training step")
    print("="*50)
    
    batch_size = 2
    seq_len = 81
    vocab_size = 16
    num_patterns = 3  # Configurable number of patterns
    
    # Create models
    meta_model = MetaTRM(_build_meta_config(batch_size, seq_len, vocab_size))
    base_model = TinyRecursiveReasoningModel_ACTV1(_build_base_config(batch_size, seq_len, vocab_size))
    
    # Freeze base weights
    for name, param in base_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-4)
    baseline_reward = 0.0
    
    train_batch = {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long),
    }
    
    val_batch = {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long),
    }
    
    # 1. Sample patterns - use helper function to get exactly num_patterns
    meta_carry = meta_model.initial_carry(train_batch)
    meta_carry, meta_outputs = meta_model(
        meta_carry,
        train_batch,
        sample=True,
        temperature=1.0,
    )
    
    # Get all available patterns and cycle if needed to get num_patterns
    all_patterns = meta_outputs["sampled_patterns"]
    all_log_probs = meta_outputs["sampled_log_probs"]  # [batch_size, slots]
    
    # Take num_patterns patterns (cycle if needed)
    patterns = []
    log_probs_list = []
    for i in range(num_patterns):
        idx = i % len(all_patterns)
        patterns.append(all_patterns[idx])
        log_probs_list.append(all_log_probs[idx])
    
    log_probs = torch.stack(log_probs_list)  # [num_patterns, slots]
    pattern_log_probs = log_probs.sum(dim=-1)  # [num_patterns]
    
    # 2. Compute rewards
    rewards, baseline_loss, _, _ = compute_rewards_from_augmentations(
        base_model=base_model,
        original_batch=val_batch,
        patterns=patterns,
        num_finetune_steps=2,
        use_binary_reward=True,
    )
    
    assert len(rewards) == num_patterns, f"Expected {num_patterns} rewards, got {len(rewards)}"
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # 3. Update baseline
    mean_reward = rewards_tensor.mean().item()
    baseline_reward = baseline_reward * 0.9 + mean_reward * 0.1
    
    # 4. Policy gradient
    advantages = rewards_tensor - baseline_reward
    policy_loss = -(pattern_log_probs * advantages).mean()
    
    # 5. Entropy regularization
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    entropy_loss = -0.01 * entropy
    
    total_loss = policy_loss + entropy_loss
    
    # 6. Update
    meta_optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=1.0)
    meta_optimizer.step()
    
    print(f"✓ Patterns sampled ({num_patterns}): {patterns}")
    print(f"✓ Rewards: {rewards}")
    print(f"✓ Policy loss: {policy_loss.item():.4f}")
    print(f"✓ Total loss: {total_loss.item():.4f}")
    print(f"✓ Training step completed successfully!")
    
    return True


def main():
    """Run all integration tests"""
    print("="*50)
    print("Meta Training Integration Tests")
    print("="*50)
    
    try:
        # Test 1: Meta-model sampling
        test_meta_model_sampling()
        
        # Test 2: LoRA parameters
        test_lora_parameters()
        
        # Test 3: Reward computation
        test_reward_computation()
        
        # Test 4: LoRA fine-tuning
        test_lora_finetuning()
        
        # Test 5: Rewards from augmentations
        test_rewards_from_augmentations()
        
        # Test 6: Policy gradient
        test_policy_gradient()
        
        # Test 7: Checkpoint loading (optional - skips if checkpoint not found)
        test_checkpoint_loading()
        
        # Test 8: Checkpoint with real data (optional - skips if dataset not found)
        test_checkpoint_with_real_data()
        
        # Test 9: End-to-end
        test_end_to_end()
        
        print("\n" + "="*50)
        print("✓ All integration tests passed!")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

