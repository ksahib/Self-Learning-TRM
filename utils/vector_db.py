"""
Vector database for puzzle similarity search.
Stores puzzle embeddings and provides fast similarity search functionality.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict
import torch


class PuzzleVectorDB:
    """
    Vector database for storing and querying puzzle embeddings.
    Uses cosine similarity for search.
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        puzzle_indices: np.ndarray,
        puzzle_identifiers: np.ndarray,
        metadata: Dict,
    ):
        """
        Initialize vector database.
        
        Args:
            embeddings: Puzzle embeddings [num_puzzles, embedding_dim]
            puzzle_indices: Dataset indices for each puzzle [num_puzzles]
            puzzle_identifiers: Puzzle IDs for each puzzle [num_puzzles]
            metadata: Metadata dict (seq_len, vocab_size, etc.)
        """
        self.embeddings = embeddings.astype(np.float32)  # Ensure float32 for efficiency
        self.puzzle_indices = puzzle_indices
        self.puzzle_identifiers = puzzle_identifiers
        self.metadata = metadata
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
        self.embeddings_normalized = self.embeddings / norms
        
        assert len(embeddings) == len(puzzle_indices) == len(puzzle_identifiers), \
            "All arrays must have same length"
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        exclude_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k most similar puzzles.
        
        Args:
            query_embedding: Query embedding [embedding_dim]
            k: Number of results to return
            exclude_indices: List of puzzle indices to exclude from results
        
        Returns:
            Tuple of (similar_puzzle_indices, similarity_scores)
            - similar_puzzle_indices: Dataset indices of similar puzzles [k]
            - similarity_scores: Cosine similarity scores [k]
        """
        query_embedding = query_embedding.astype(np.float32).flatten()
        
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        else:
            query_embedding = query_embedding
        
        # Compute cosine similarity (dot product with normalized embeddings)
        similarities = np.dot(self.embeddings_normalized, query_embedding)
        
        # Exclude specified indices
        if exclude_indices is not None:
            exclude_mask = np.zeros(len(similarities), dtype=bool)
            for idx in exclude_indices:
                # Find puzzle_indices that match
                mask = self.puzzle_indices == idx
                exclude_mask |= mask
            similarities[exclude_mask] = -np.inf
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # Filter out -inf (excluded items)
        valid_mask = similarities[top_k_indices] > -np.inf
        top_k_indices = top_k_indices[valid_mask]
        
        # Return dataset indices and scores
        similar_puzzle_indices = self.puzzle_indices[top_k_indices]
        similarity_scores = similarities[top_k_indices]
        
        return similar_puzzle_indices, similarity_scores
    
    def get_embedding_by_index(self, puzzle_index: int) -> Optional[np.ndarray]:
        """Get embedding for a specific puzzle index."""
        mask = self.puzzle_indices == puzzle_index
        if np.any(mask):
            idx = np.where(mask)[0][0]
            return self.embeddings[idx]
        return None
    
    def save(self, path: str):
        """Save vector database to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            puzzle_indices=self.puzzle_indices,
            puzzle_identifiers=self.puzzle_identifiers,
            metadata=self.metadata,
        )
        print(f"Saved vector database to {path}")
    
    @classmethod
    def load(cls, path: str) -> "PuzzleVectorDB":
        """Load vector database from disk."""
        data = np.load(path, allow_pickle=True)
        
        # Handle metadata (may be stored as object array)
        metadata = data["metadata"]
        if isinstance(metadata, np.ndarray) and metadata.dtype == object:
            metadata = metadata.item()
        
        return cls(
            embeddings=data["embeddings"],
            puzzle_indices=data["puzzle_indices"],
            puzzle_identifiers=data["puzzle_identifiers"],
            metadata=metadata,
        )
    
    def __len__(self):
        return len(self.embeddings)

