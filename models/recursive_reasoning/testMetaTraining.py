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
    
    print(f"✓ Computed rewards for {len(patterns)} patterns: {rewards}")
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


def test_end_to_end():
    """Test 7: End-to-end training step"""
    print("\n" + "="*50)
    print("Test 7: End-to-end training step")
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
        
        # Test 7: End-to-end
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

