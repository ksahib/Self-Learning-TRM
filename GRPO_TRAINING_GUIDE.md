# GRPO Two-Stage Training Guide

This guide explains how to use the new two-stage GRPO training approach:
1. **Stage 1**: PPO-style GRPO (simple, no reference model)
2. **Stage 2**: Reference-model GRPO (more stable for fine-tuning)

## Quick Start

### Stage 1: PPO-Style GRPO (Recommended for Initial Training)

```bash
python meta_train.py \
    data_paths="[data/sudoku-extreme-250-aug-250]" \
    data_paths_test="[data/sudoku-extreme-250-aug-250]" \
    meta_epochs=20000 \
    eval_interval=200 \
    checkpoint_path="/kaggle/working/checkpoints/MetaTRM-REINFORCE/kaggle-run-3" \
    load_checkpoint="{latest_checkpoint_path}" \
    checkpoint_interval=100 \
    use_reference_model=false
```

**Key points:**
- `use_reference_model=false` → Uses PPO-style (old_log_probs from sampling)
- No reference model needed → Simpler, faster, avoids initialization issues
- Works great for initial training

### Stage 2: Reference-Model GRPO (For Fine-Tuning)

After training with PPO-style, you can switch to reference-model GRPO:

```bash
python meta_train.py \
    data_paths="[data/sudoku-extreme-250-aug-250]" \
    data_paths_test="[data/sudoku-extreme-250-aug-250]" \
    meta_epochs=20000 \
    eval_interval=200 \
    checkpoint_path="/kaggle/working/checkpoints/MetaTRM-REINFORCE/kaggle-run-4" \
    load_checkpoint="/kaggle/working/checkpoints/MetaTRM-REINFORCE/kaggle-run-3/meta_step_1000.pt" \
    checkpoint_interval=100 \
    use_reference_model=true \
    reference_checkpoint="/kaggle/working/checkpoints/MetaTRM-REINFORCE/kaggle-run-3/meta_step_1000.pt"
```

**Key points:**
- `use_reference_model=true` → Uses reference model GRPO
- `reference_checkpoint` → Load reference model from your trained checkpoint (stage 1)
- `load_checkpoint` → Continue training from where you left off
- More stable for fine-tuning

## Auto-Switch Mode

You can also train in one run and auto-switch at a specific step:

```bash
python meta_train.py \
    data_paths="[data/sudoku-extreme-250-aug-250]" \
    data_paths_test="[data/sudoku-extreme-250-aug-250]" \
    meta_epochs=20000 \
    eval_interval=200 \
    checkpoint_path="/kaggle/working/checkpoints/MetaTRM-REINFORCE/kaggle-run-5" \
    load_checkpoint="{latest_checkpoint_path}" \
    checkpoint_interval=100 \
    use_reference_model=true \
    switch_to_reference_at_step=1000
```

**Key points:**
- Starts with PPO-style (use_reference_model=true but switch_to_reference_at_step=1000)
- Automatically switches to reference-model GRPO at step 1000
- Reference model is initialized from current model at step 1000

## Configuration Options

### New Config Parameters

- `use_reference_model: bool = False`
  - If `False`: Use PPO-style GRPO (old_log_probs)
  - If `True`: Use reference-model GRPO
  
- `reference_checkpoint: Optional[str] = None`
  - Path to checkpoint to load as reference model
  - If `None` and `use_reference_model=True`: Creates deep copy of current model
  
- `switch_to_reference_at_step: Optional[int] = None`
  - Auto-switch to reference model at this step
  - If `None`: Manual control (use `use_reference_model` flag)

### Existing Config (Still Works)

- `ppo_clip_epsilon: float = 0.2` - PPO clipping epsilon
- `use_kl_penalty: bool = True` - Use KL penalty (only if using reference model)
- `kl_target: float = 0.01` - Target KL divergence
- `use_ref_ema: bool = True` - Use EMA for reference model updates
- `ref_ema_decay: float = 0.995` - EMA decay rate
- `ref_update_interval: int = 500` - Hard copy interval (if not using EMA)

## Training Workflow

### Recommended Workflow

1. **Initial Training (Stage 1)**
   ```bash
   # Train with PPO-style (simple, stable)
   use_reference_model=false
   ```

2. **Save Checkpoint**
   - After some training (e.g., 1000 steps)
   - Checkpoint saved at: `checkpoint_path/meta_step_1000.pt`

3. **Fine-Tuning (Stage 2)**
   ```bash
   # Continue with reference-model GRPO
   use_reference_model=true
   reference_checkpoint="path/to/meta_step_1000.pt"
   load_checkpoint="path/to/meta_step_1000.pt"
   ```

## Monitoring

Check `meta/using_reference_model` in wandb:
- `0.0` = PPO-style mode
- `1.0` = Reference-model mode

## Troubleshooting

### If reference model fails:
- The code will automatically fall back to PPO-style on first step
- Or use `use_reference_model=false` to avoid issues entirely

### If you want to switch modes mid-training:
- Save a checkpoint
- Restart with different `use_reference_model` setting
- Load checkpoint with `load_checkpoint`

## Benefits

1. **Stage 1 (PPO-style)**: Simple, fast, avoids initialization issues
2. **Stage 2 (Reference-model)**: More stable anchor for fine-tuning
3. **Flexibility**: Switch between modes as needed
4. **Backward Compatible**: Old configs still work (defaults to PPO-style)

