"""
Few-shot dataset wrapper that uses vector database for similarity-based sampling.
Replaces random batch sampling with similarity-based retrieval.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from torch.utils.data import IterableDataset

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.vector_db import PuzzleVectorDB
from models.recursive_reasoning.helper import apply_augmentation_patterns_to_batch
from models.recursive_reasoning.metaTRM import MetaTRM
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


class FewShotDataset(IterableDataset):
    """
    Dataset wrapper that implements SEAL-style few-shot prompting.
    For each puzzle, finds similar puzzles via vector database,
    fetches them, and applies MetaTRM-selected augmentations.
    Original puzzles are never augmented.
    """
    
    def __init__(
        self,
        base_dataset: PuzzleDataset,
        vector_db: PuzzleVectorDB,
        meta_model: MetaTRM,
        base_model: TinyRecursiveReasoningModel_ACTV1,
        num_similar_examples: int,
        device: str = "cuda",
        grid_height: int = 9,
        grid_width: int = 9,
    ):
        """
        Initialize few-shot dataset.
        
        Args:
            base_dataset: Base PuzzleDataset to wrap
            vector_db: Vector database for similarity search
            meta_model: MetaTRM model for augmentation selection
            base_model: Base TRM model for encoding queries
            num_similar_examples: Number of similar examples to fetch per puzzle
            device: Device to run models on
            grid_height: Height of grid (default 9 for Sudoku)
            grid_width: Width of grid (default 9 for Sudoku)
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.vector_db = vector_db
        self.meta_model = meta_model
        self.base_model = base_model
        self.num_similar_examples = num_similar_examples
        self.device = device
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        self.metadata = base_dataset.metadata
        self.config = base_dataset.config
        
        # Store device but don't move models here (will be moved when needed)
        # Models will be moved to device on-demand to avoid CUDA issues in worker processes
        self.base_model.eval()
        self.meta_model.eval()
    
    def _encode_puzzle(self, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor) -> np.ndarray:
        """Encode a puzzle to get its embedding."""
        # Move models and inputs to device on-demand (not in __init__ to avoid CUDA in workers)
        base_model = self.base_model.to(self.device)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            puzzle_identifiers = puzzle_identifiers.to(self.device)
            
            # Get input embeddings
            input_embeddings = base_model.inner._input_embeddings(inputs, puzzle_identifiers)
            
            # Mean pool to get fixed-size embedding
            embedding = input_embeddings.mean(dim=1)  # [batch_size, hidden_size]
            
            # Convert to numpy (cast to float32 first to handle bfloat16)
            embedding = embedding.to(dtype=torch.float32).cpu()
            return embedding.numpy()
    
    def _get_similar_examples(
        self,
        query_embedding: np.ndarray,
        query_puzzle_identifier: int,
        dataset_data: Dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get similar examples from dataset.
        
        Args:
            query_embedding: Embedding of query puzzle
            query_puzzle_identifier: Puzzle identifier to exclude from results
            dataset_data: Dataset data dict
        
        Returns:
            Tuple of (similar_inputs, similar_labels, similar_puzzle_identifiers)
        """
        # Search vector database
        # Exclude puzzle by identifier (vector_db stores puzzle_identifiers)
        similar_puzzle_indices, _ = self.vector_db.search(
            query_embedding,
            k=self.num_similar_examples + 1,  # +1 to account for excluding self
            exclude_indices=None,  # We'll filter by identifier instead
        )
        
        # Get puzzle identifiers for similar puzzles
        similar_puzzle_ids = []
        for idx in similar_puzzle_indices:
            # Find puzzle identifier in vector_db
            mask = self.vector_db.puzzle_indices == idx
            if np.any(mask):
                puzzle_id = self.vector_db.puzzle_identifiers[np.where(mask)[0][0]]
                if puzzle_id != query_puzzle_identifier:  # Exclude self
                    similar_puzzle_ids.append((idx, puzzle_id))
        
        # Limit to num_similar_examples
        similar_puzzle_ids = similar_puzzle_ids[:self.num_similar_examples]
        
        # Fetch examples from dataset
        similar_inputs_list = []
        similar_labels_list = []
        similar_puzzle_identifiers_list = []
        
        for set_name, dataset in dataset_data.items():
            # Find examples with matching puzzle identifiers
            for puzzle_idx, puzzle_id in similar_puzzle_ids:
                # Find puzzle by identifier
                puzzle_mask = dataset["puzzle_identifiers"] == puzzle_id
                matching_puzzles = np.where(puzzle_mask)[0]
                
                if len(matching_puzzles) > 0:
                    # Use first matching puzzle
                    puzzle_idx_in_dataset = matching_puzzles[0]
                    
                    # Find example indices for this puzzle
                    puzzle_start = dataset["puzzle_indices"][puzzle_idx_in_dataset]
                    puzzle_end = dataset["puzzle_indices"][puzzle_idx_in_dataset + 1] if puzzle_idx_in_dataset + 1 < len(dataset["puzzle_indices"]) else len(dataset["inputs"])
                    
                    # Get a random example from this puzzle
                    if puzzle_end > puzzle_start:
                        example_idx = np.random.randint(puzzle_start, puzzle_end)
                        similar_inputs_list.append(dataset["inputs"][example_idx])
                        similar_labels_list.append(dataset["labels"][example_idx])
                        similar_puzzle_identifiers_list.append(puzzle_id)
        
        if len(similar_inputs_list) == 0:
            # Fallback: return empty arrays with correct shape
            seq_len = self.metadata.seq_len
            similar_inputs = np.zeros((0, seq_len), dtype=np.int32)
            similar_labels = np.zeros((0, seq_len), dtype=np.int32)
            similar_puzzle_identifiers = np.zeros((0,), dtype=np.int32)
        else:
            similar_inputs = np.stack(similar_inputs_list)
            similar_labels = np.stack(similar_labels_list)
            similar_puzzle_identifiers = np.array(similar_puzzle_identifiers_list)
        
        return similar_inputs, similar_labels, similar_puzzle_identifiers
    
    def _apply_metatrm_augmentations(
        self,
        similar_inputs: torch.Tensor,
        similar_labels: torch.Tensor,
        similar_puzzle_identifiers: torch.Tensor,
        meta_model=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use MetaTRM to select and apply augmentations to similar examples.
        
        Args:
            similar_inputs: Input tensors (already on device)
            similar_labels: Label tensors (already on device)
            similar_puzzle_identifiers: Puzzle identifier tensors (already on device)
            meta_model: MetaTRM model (if None, uses self.meta_model)
        
        Returns:
            Tuple of (augmented_inputs, augmented_labels)
        """
        # Handle empty batch
        if len(similar_inputs) == 0:
            return similar_inputs, similar_labels
        
        # Use provided meta_model or fall back to self.meta_model
        if meta_model is None:
            meta_model = self.meta_model.to(self.device)
        
        # Create batch dict for MetaTRM
        batch = {
            "inputs": similar_inputs,
            "labels": similar_labels,
            "puzzle_identifiers": similar_puzzle_identifiers,
        }
        
        # Get augmentation patterns from MetaTRM
        meta_carry = meta_model.initial_carry(batch)
        meta_carry, meta_outputs = meta_model(
            meta_carry,
            batch,
            sample=True,
            temperature=1.0,
        )
        
        patterns = meta_outputs["sampled_patterns"]  # List[str]
        
        # Handle case where patterns might be empty
        if len(patterns) == 0:
            return similar_inputs, similar_labels
        
        # Apply augmentations
        aug_batch = apply_augmentation_patterns_to_batch(
            batch,
            patterns,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )
        
        return aug_batch["inputs"], aug_batch["labels"]
    
    def _collate_batch(self, batch: Dict) -> Dict:
        """Collate batch similar to PuzzleDataset."""
        # Convert dtype
        batch = {k: v.astype(np.int32) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
        
        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = -100  # IGNORE_LABEL_ID
        
        # Pad if needed
        if batch["puzzle_identifiers"].size < self.config.global_batch_size:
            pad_size = self.config.global_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": -100,  # IGNORE_LABEL_ID
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {
                k: np.pad(v, ((0, pad_size),) + ((0, 0),) * (v.ndim - 1), constant_values=pad_values.get(k, 0))
                for k, v in batch.items()
            }
        
        # To tensor
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
    
    def __iter__(self):
        """Iterate over few-shot batches."""
        # Lazy load dataset
        self.base_dataset._lazy_load_dataset()
        dataset_data = self.base_dataset._data
        
        # Iterate over base dataset
        for set_name, batch, global_batch_size in self.base_dataset:
            # batch contains: inputs, labels, puzzle_identifiers
            batch_size = batch["inputs"].shape[0]
            
            # Prepare output batch
            original_inputs = batch["inputs"].cpu().numpy()
            original_labels = batch["labels"].cpu().numpy()
            original_puzzle_identifiers = batch["puzzle_identifiers"].cpu().numpy()
            
            # Lists to collect similar examples
            all_similar_inputs = []
            all_similar_labels = []
            all_similar_puzzle_ids = []
            
            # Process each puzzle in batch
            # Move models to device on-demand (safe with num_workers=0)
            base_model = self.base_model.to(self.device)
            meta_model = self.meta_model.to(self.device)
            
            for i in range(batch_size):
                # Get original puzzle (never augmented)
                orig_input = torch.from_numpy(original_inputs[i:i+1]).to(self.device)
                orig_puzzle_id = torch.from_numpy(original_puzzle_identifiers[i:i+1]).to(self.device)
                
                # Encode original puzzle
                query_embedding = self._encode_puzzle(orig_input, orig_puzzle_id)[0]
                
                # Get puzzle identifier
                query_puzzle_identifier = int(original_puzzle_identifiers[i])
                
                # Get similar examples
                similar_inputs_np, similar_labels_np, similar_puzzle_ids_np = self._get_similar_examples(
                    query_embedding,
                    query_puzzle_identifier,
                    dataset_data,
                )
                
                # Convert to tensors
                similar_inputs = torch.from_numpy(similar_inputs_np).to(self.device)
                similar_labels = torch.from_numpy(similar_labels_np).to(self.device)
                # Use actual puzzle identifiers from similar examples (or fallback to query puzzle ID if empty)
                if len(similar_puzzle_ids_np) > 0:
                    similar_puzzle_ids = torch.from_numpy(similar_puzzle_ids_np).to(self.device)
                else:
                    # Fallback: use query puzzle identifier if no similar examples found
                    similar_puzzle_ids = torch.full((0,), query_puzzle_identifier, dtype=torch.long, device=self.device)
                
                # Apply MetaTRM augmentations
                if len(similar_inputs) > 0:
                    aug_inputs, aug_labels = self._apply_metatrm_augmentations(
                        similar_inputs,
                        similar_labels,
                        similar_puzzle_ids,
                        meta_model=meta_model,
                    )
                    all_similar_inputs.append(aug_inputs.cpu().numpy())
                    all_similar_labels.append(aug_labels.cpu().numpy())
                    # Store puzzle identifiers for augmented examples
                    all_similar_puzzle_ids.append(similar_puzzle_ids.cpu().numpy())
                else:
                    # Empty similar examples
                    all_similar_inputs.append(np.zeros((0, self.metadata.seq_len), dtype=np.int32))
                    all_similar_labels.append(np.zeros((0, self.metadata.seq_len), dtype=np.int32))
                    all_similar_puzzle_ids.append(np.zeros((0,), dtype=np.int32))
            
            # Stack similar examples
            # Pad to same number of examples per puzzle
            max_similar = max(len(sim) for sim in all_similar_inputs) if all_similar_inputs else 0
            
            if max_similar > 0:
                # Pad and stack
                padded_similar_inputs = []
                padded_similar_labels = []
                padded_similar_puzzle_ids = []
                
                for sim_inputs, sim_labels, sim_puzzle_ids in zip(all_similar_inputs, all_similar_labels, all_similar_puzzle_ids):
                    if len(sim_inputs) < max_similar:
                        pad_size = max_similar - len(sim_inputs)
                        sim_inputs = np.pad(sim_inputs, ((0, pad_size), (0, 0)), constant_values=self.metadata.pad_id)
                        sim_labels = np.pad(sim_labels, ((0, pad_size), (0, 0)), constant_values=-100)
                        # Pad puzzle identifiers with the last valid identifier (or 0 if empty)
                        pad_value = int(sim_puzzle_ids[-1]) if len(sim_puzzle_ids) > 0 else 0
                        sim_puzzle_ids = np.pad(sim_puzzle_ids, (0, pad_size), constant_values=pad_value)
                    padded_similar_inputs.append(sim_inputs)
                    padded_similar_labels.append(sim_labels)
                    padded_similar_puzzle_ids.append(sim_puzzle_ids)
                
                similar_inputs_stacked = np.stack(padded_similar_inputs)  # [batch_size, max_similar, seq_len]
                similar_labels_stacked = np.stack(padded_similar_labels)  # [batch_size, max_similar, seq_len]
                similar_puzzle_ids_stacked = np.stack(padded_similar_puzzle_ids)  # [batch_size, max_similar]
            else:
                # No similar examples
                similar_inputs_stacked = np.zeros((batch_size, 0, self.metadata.seq_len), dtype=np.int32)
                similar_labels_stacked = np.zeros((batch_size, 0, self.metadata.seq_len), dtype=np.int32)
                similar_puzzle_ids_stacked = np.zeros((batch_size, 0), dtype=np.int32)
            
            # Create output batch
            output_batch = {
                "inputs": original_inputs,  # Original puzzles (never augmented)
                "labels": original_labels,  # Original labels (will be masked in loss)
                "similar_inputs": similar_inputs_stacked,  # Augmented similar examples
                "similar_labels": similar_labels_stacked,  # Augmented similar labels
                "similar_puzzle_identifiers": similar_puzzle_ids_stacked,  # Puzzle identifiers for similar examples
                "puzzle_identifiers": original_puzzle_identifiers,
                "is_original": np.ones(batch_size, dtype=bool),  # All are originals
            }
            
            # Collate
            output_batch = self._collate_batch(output_batch)
            
            yield set_name, output_batch, global_batch_size

