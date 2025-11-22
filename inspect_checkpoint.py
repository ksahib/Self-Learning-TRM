#!/usr/bin/env python3
"""Inspect checkpoint contents to see what keys are present."""
import torch

checkpoint_path = "step_21700"
print(f"Inspecting checkpoint: {checkpoint_path}\n")

ckpt = torch.load(checkpoint_path, map_location="cpu")
keys = list(ckpt.keys())

print(f"Total keys: {len(keys)}\n")
print("=" * 80)
print("First 20 keys:")
print("=" * 80)
for k in keys[:20]:
    print(f"  {k}")

print("\n" + "=" * 80)
print("Keys containing 'lora':")
print("=" * 80)
lora_keys = [k for k in keys if 'lora' in k.lower()]
if lora_keys:
    for k in lora_keys:
        print(f"  {k}")
else:
    print("  (none found)")

print("\n" + "=" * 80)
print("Keys containing 'lm_head':")
print("=" * 80)
lm_head_keys = [k for k in keys if 'lm_head' in k]
for k in lm_head_keys:
    print(f"  {k}")

print("\n" + "=" * 80)
print("Keys containing 'q_head':")
print("=" * 80)
q_head_keys = [k for k in keys if 'q_head' in k]
for k in q_head_keys:
    print(f"  {k}")

print("\n" + "=" * 80)
print("Keys containing 'base':")
print("=" * 80)
base_keys = [k for k in keys if '.base.' in k]
if base_keys:
    for k in base_keys[:10]:
        print(f"  {k}")
    if len(base_keys) > 10:
        print(f"  ... and {len(base_keys) - 10} more")
else:
    print("  (none found)")

print("\n" + "=" * 80)
print("Sample of all keys (showing structure):")
print("=" * 80)
for k in keys[:15]:
    print(f"  {k}")

