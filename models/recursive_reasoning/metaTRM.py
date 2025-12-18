from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.mlayers import (
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)
from models.sparse_embedding import CastedSparseEmbedding
from models.recursive_reasoning.helper import (
    greedy_indices_from_logits,
    sample_indices_from_logits,
    indices_to_grid_tensor,
    indices_to_pattern_strings,
)
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


@dataclass
class MetaTRMInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class MetaTRMCarry:
    inner_carry: MetaTRMInnerCarry
    current_data: Dict[str, torch.Tensor]


class MetaTRMConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    # Output grid config
    aug_slots: int = 6   # 3x2 grid
    choices_per_slot: int = 2

    # Allowed cycle choices for base TRM (discrete sets, enforce bounds here)
    # H_cycles ∈ h_cycle_choices, e.g., [1, 2, 3]
    # L_cycles ∈ l_cycle_choices, e.g., [1, 2, 3, 4, 5, 6]
    h_cycle_choices: List[int] = [1, 2, 3]
    l_cycle_choices: List[int] = [1, 2, 3, 4, 5, 6]

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False
    puzzle_emb_len: int = 16


class MetaTRMBlock(nn.Module):
    def __init__(self, config: MetaTRMConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = (
                -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
                if self.config.puzzle_emb_len == 0
                else self.config.puzzle_emb_len
            )
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class MetaTRMReasoningModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs):
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class MetaTRMInner(nn.Module):
    def __init__(self, config: MetaTRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )

        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0
            else self.config.puzzle_emb_len
        )
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )

        self.L_level = MetaTRMReasoningModule(
            layers=[MetaTRMBlock(self.config) for _ in range(self.config.L_layers)]
        )

        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        self.aug_queries = nn.Parameter(
            trunc_normal_init_(
                torch.empty(self.config.aug_slots, self.config.hidden_size, dtype=self.forward_dtype), std=1
            )
        )
        self.aug_pool = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_heads,
            batch_first=True,
        )
        self.aug_head = CastedLinear(
            self.config.hidden_size,
            self.config.choices_per_slot,
            bias=False,
        )
        # Heads for H/L cycle choices (operate on pooled hidden state)
        self.h_head = CastedLinear(
            self.config.hidden_size,
            len(self.config.h_cycle_choices),
            b…

    def _input_embeddings(self, inputs: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor]):
        embedding = self.embed_tokens(inputs.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0 and puzzle_identifiers is not None:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            puzzle_embedding = puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size)
            embedding = torch.cat((puzzle_embedding, embedding), dim=-2)

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        device = next(self.parameters()).device
        return MetaTRMInnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def forward(
        self,
        carry: MetaTRMInnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[MetaTRMInnerCarry, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch.get("puzzle_identifiers"),
        )

        z_H, z_L = carry.z_H, carry.z_L

        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        new_carry = MetaTRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())

        attn_dtype = self.aug_pool.in_proj_weight.dtype
        key_value = z_H[:, self.puzzle_emb_len :].to(attn_dtype)
        aug_queries = (
            self.aug_queries.unsqueeze(0).expand(key_value.shape[0], -1, -1).to(attn_dtype)
        )
        aug_repr, _ = self.aug_pool(query=aug_queries, key=key_value, value=key_value)
        aug_logits = self.aug_head(aug_repr)

        # Pool across augmentation slots to get a single representation per puzzle
        pooled = aug_repr.mean(dim=1)  # [batch, hidden_size]
        h_logits = self.h_head(pooled)
        l_logits = self.l_head(pooled)

        return new_carry, aug_logits, h_logits, l_logits


class MetaTRM(nn.Module):
    """
    Meta-level controller that proposes augmentation grids and can interface with the
    base TinyRecursiveReasoningModel.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = MetaTRMConfig(**config_dict)
        self.inner = MetaTRMInner(self.config)

    @property
    def puzzle_emb(self):
        return getattr(self.inner, "puzzle_emb", None)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> MetaTRMCarry:
        batch_size = batch["inputs"].shape[0]
        # Handle empty batch
        if batch_size == 0:
            # Return empty carry with correct structure
            device = batch["inputs"].device
            return MetaTRMCarry(
                inner_carry=self.inner.empty_carry(1),  # Use 1 for empty batch structure
                current_data={k: torch.empty((0, *v.shape[1:]), dtype=v.dtype, device=device) for k, v in batch.items()},
            )
        return MetaTRMCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: MetaTRMCarry,
        batch: Dict[str, torch.Tensor],
        *,
        sample: bool = False,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Tuple[MetaTRMCarry, Dict[str, torch.Tensor]]:
        # Handle empty batch
        batch_size = batch["inputs"].shape[0]
        if batch_size == 0:
            # Return empty outputs
            device = batch["inputs"].device
            empty_aug_logits = torch.empty(
                (0, self.config.aug_slots, self.config.choices_per_slot),
                device=device,
                dtype=self.inner.forward_dtype,
            )
            empty_h_logits = torch.empty(
                (0, len(self.config.h_cycle_choices)),
                device=device,
                dtype=self.inner.forward_dtype,
            )
            empty_l_logits = torch.empty(
                (0, len(self.config.l_cycle_choices)),
                device=device,
                dtype=self.inner.forward_dtype,
            )
            outputs: Dict[str, torch.Tensor] = {
                "aug_logits": empty_aug_logits,
                "h_logits": empty_h_logits,
                "l_logits": empty_l_logits,
            }
            outputs["sampled_patterns"] = []
            outputs["sampled_H_values"] = torch.empty((0,), device=device, dtype=torch.long)
            outputs["sampled_L_values"] = torch.empty((0,), device=device, dtype=torch.long)
            outputs["sampled_H_log_probs"] = torch.empty(
                (0,), device=device, dtype=self.inner.forward_dtype
            )
            outputs["sampled_L_log_probs"] = torch.empty(
                (0,), device=device, dtype=self.inner.forward_dtype
            )
            return MetaTRMCarry(carry.inner_carry, batch), outputs
        
        new_current_data = {k: batch[k] for k in batch}
        new_inner_carry, aug_logits, h_logits, l_logits = self.inner(
            carry.inner_carry, new_current_data
        )

        outputs: Dict[str, torch.Tensor] = {
            "aug_logits": aug_logits,
            "h_logits": h_logits,
            "l_logits": l_logits,
        }

        if greedy:
            indices = greedy_indices_from_logits(aug_logits)
            outputs["greedy_indices"] = indices
            outputs["greedy_grid"] = indices_to_grid_tensor(indices)
            outputs["greedy_patterns"] = indices_to_pattern_strings(indices)
        elif sample:
            sampled_indices, slot_log_probs = sample_indices_from_logits(
                aug_logits, temperature=temperature
            )
            outputs["sampled_indices"] = sampled_indices
            outputs["sampled_log_probs"] = slot_log_probs
            outputs["sampled_grid"] = indices_to_grid_tensor(sampled_indices)
            outputs["sampled_patterns"] = indices_to_pattern_strings(sampled_indices)

            # Sample H-cycles
            h_probs = torch.softmax(h_logits, dim=-1)
            h_probs = torch.nan_to_num(
                h_probs,
                nan=1.0 / max(h_logits.shape[-1], 1),
                posinf=1.0,
                neginf=0.0,
            )
            h_probs = h_probs / h_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            h_indices = torch.multinomial(h_probs, num_samples=1).squeeze(-1)
            h_log_probs = torch.log(
                h_probs.gather(dim=-1, index=h_indices.unsqueeze(-1))
                .squeeze(-1)
                .clamp_min(1e-8)
            )

            # Sample L-cycles
            l_probs = torch.softmax(l_logits, dim=-1)
            l_probs = torch.nan_to_num(
                l_probs,
                nan=1.0 / max(l_logits.shape[-1], 1),
                posinf=1.0,
                neginf=0.0,
            )
            l_probs = l_probs / l_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            l_indices = torch.multinomial(l_probs, num_samples=1).squeeze(-1)
            l_log_probs = torch.log(
                l_probs.gather(dim=-1, index=l_indices.unsqueeze(-1))
                .squeeze(-1)
                .clamp_min(1e-8)
            )

            # Map indices to concrete H/L values
            h_choices = torch.tensor(
                self.config.h_cycle_choices,
                device=h_indices.device,
                dtype=torch.long,
            )
            l_choices = torch.tensor(
                self.config.l_cycle_choices,
                device=l_indices.device,
                dtype=torch.long,
            )
            sampled_H_values = h_choices[h_indices]
            sampled_L_values = l_choices[l_indices]

            outputs["sampled_H_indices"] = h_indices
            outputs["sampled_L_indices"] = l_indices
            outputs["sampled_H_values"] = sampled_H_values
            outputs["sampled_L_values"] = sampled_L_values
            outputs["sampled_H_log_probs"] = h_log_probs
            outputs["sampled_L_log_probs"] = l_log_probs

        return MetaTRMCarry(new_inner_carry, new_current_data), outputs

    # ------------------------------------------------------------------
    # Base TRM integration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def build_base_trm(config_dict: dict) -> TinyRecursiveReasoningModel_ACTV1:
        """
        Convenience helper to instantiate the base TinyRecursiveReasoningModel.
        """
        return TinyRecursiveReasoningModel_ACTV1(config_dict)

    @staticmethod
    def run_base_trm_step(
        base_model: TinyRecursiveReasoningModel_ACTV1,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Execute a single forward pass of the base TRM (without loss head).
        """
        carry = base_model.initial_carry(batch)
        return base_model(carry=carry, batch=batch)

    @staticmethod
    def execute_base_trm_with_augmentations(
        base_model: TinyRecursiveReasoningModel_ACTV1,
        batch: Dict[str, torch.Tensor],
        patterns: List[str],
        num_finetune_steps: int = 0,
        grid_height: int = 9,
        grid_width: int = 9,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Apply augmentations and run base TRM on each augmented batch.
        
        Args:
            base_model: Base TRM model instance
            batch: Original batch with 'inputs', 'labels', etc.
            patterns: List of augmentation pattern strings (one per puzzle in batch)
            num_finetune_steps: Number of fine-tuning steps to run before evaluation.
                              If 0, just evaluates without fine-tuning.
            grid_height: Height of grid (default 9 for Sudoku)
            grid_width: Width of grid (default 9 for Sudoku)
        
        Returns:
            List of base TRM outputs (one dict per augmentation pattern).
            Each dict contains 'logits', 'q_halt_logits', 'q_continue_logits', etc.
        """
        from models.recursive_reasoning.helper import apply_augmentation_patterns_to_batch
        
        base_outputs_list = []
        
        # Apply each pattern and run base TRM
        for pattern in patterns:
            # Apply augmentation to batch
            aug_batch = apply_augmentation_patterns_to_batch(
                batch, [pattern], grid_height=grid_height, grid_width=grid_width
            )
            
            # Fine-tune if requested (simple gradient steps)
            if num_finetune_steps > 0:
                base_model.train()
                # Simple optimizer for base model parameters
                optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-4)
                
                for _ in range(num_finetune_steps):
                    carry = base_model.initial_carry(aug_batch)
                    carry, outputs = base_model(carry=carry, batch=aug_batch)
                    
                    # Simple loss: cross-entropy on logits if labels exist
                    if "labels" in aug_batch:
                        logits = outputs["logits"]
                        labels = aug_batch["labels"]
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.shape[-1]),
                            labels.view(-1),
                            ignore_index=-100
                        )
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
            
            # Evaluate on augmented batch
            base_model.eval()
            with torch.no_grad():
                carry = base_model.initial_carry(aug_batch)
                carry, outputs = base_model(carry=carry, batch=aug_batch)
                base_outputs_list.append(outputs)
        
        return base_outputs_list

    @staticmethod
    def load_base_trm_from_checkpoint(
        checkpoint_path: str,
        config_dict: dict,
        map_location: str = "cpu",
        strict: bool = False,
    ) -> TinyRecursiveReasoningModel_ACTV1:
        """
        Build base TRM and load pretrained checkpoint.
        
        Convenience method that combines build_base_trm and load_checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config_dict: Configuration dict for base TRM
            map_location: Device to load checkpoint on
            strict: Whether to require exact key match
        
        Returns:
            Base TRM model with loaded weights
        """
        from models.recursive_reasoning.helper import load_base_trm_checkpoint
        
        # Build base TRM
        base_model = MetaTRM.build_base_trm(config_dict)
        
        # Load checkpoint
        load_base_trm_checkpoint(
            checkpoint_path=checkpoint_path,
            model=base_model,
            map_location=map_location,
            strict=strict,
        )
        
        return base_model

