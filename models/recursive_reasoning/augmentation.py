"""
Data augmentation engine for grid-based puzzles (ARC, Sudoku, Maze).
Implements discrete augmentations controlled by meta-model.
"""

from typing import Tuple, Optional
import torch
import numpy as np



# Augmentation type codes
AUG_NONE = '.'      # No change
AUG_ROT_LEFT = 'L'   # Rotate 90 degrees counter-clockwise
AUG_ROT_RIGHT = 'R'  # Rotate 90 degrees clockwise
AUG_ROT_180 = 'O'    # Rotate 180 degrees
AUG_FLIP_V = 'V'     # Flip vertical (up-down)
AUG_FLIP_H = 'H'     # Flip horizontal (left-right)
AUG_TRANSPOSE = 'T'  # Transpose (swap rows and columns)

# All augmentation types (7 total)
ALL_AUG_TYPES = [AUG_NONE, AUG_ROT_LEFT, AUG_ROT_RIGHT, AUG_ROT_180, 
                 AUG_FLIP_V, AUG_FLIP_H, AUG_TRANSPOSE]

AUG_TYPE_TO_ID = {aug: i for i, aug in enumerate(ALL_AUG_TYPES)}
AUG_ID_TO_TYPE = {i: aug for i, aug in enumerate(ALL_AUG_TYPES)}

# Position-specific binary choices for 3x2 grid
# Each position (row, col) can only be one of two values:
# Position (1,1) = row 0, col 0: '.' or 'L'
# Position (1,2) = row 0, col 1: '.' or 'R'
# Position (2,1) = row 1, col 0: '.' or 'H'
# Position (2,2) = row 1, col 1: '.' or 'V'
# Position (3,1) = row 2, col 0: '.' or 'O'
# Position (3,2) = row 2, col 1: '.' or 'T'

POSITION_TO_AUG_CHOICES = {
    (0, 0): ['.', 'L'],  # Position (1,1)
    (0, 1): ['.', 'R'],  # Position (1,2)
    (1, 0): ['.', 'H'],  # Position (2,1)
    (1, 1): ['.', 'V'],  # Position (2,2)
    (2, 0): ['.', 'O'],  # Position (3,1)
    (2, 1): ['.', 'T'],  # Position (3,2)
}

# Mapping from position index (0-5) to choices
# Position order: (0,0), (0,1), (1,0), (1,1), (2,0), (2,1)
POSITION_INDEX_TO_CHOICES = [
    ['.', 'L'],  # index 0: (0,0)
    ['.', 'R'],  # index 1: (0,1)
    ['.', 'H'],  # index 2: (1,0)
    ['.', 'V'],  # index 3: (1,1)
    ['.', 'O'],  # index 4: (2,0)
    ['.', 'T'],  # index 5: (2,1)
]

# All possible values in the pattern (for pattern parsing)
ALL_PATTERN_VALUES = ALL_AUG_TYPES


def apply_single_augmentation(grid: np.ndarray, aug_type: str) -> np.ndarray:
    """
    Apply a single augmentation to a 2D grid.
    
    Args:
        grid: 2D numpy array (H, W)
        aug_type: Augmentation type character
    
    Returns:
        Augmented grid (may have different shape)
    """
    if aug_type == AUG_NONE or aug_type == '.':
        return grid.copy()
    
    elif aug_type == AUG_ROT_LEFT or aug_type == 'L':
        # Rotate 90 degrees counter-clockwise
        return np.rot90(grid, k=1)
    
    elif aug_type == AUG_ROT_RIGHT or aug_type == 'R':
        # Rotate 90 degrees clockwise
        return np.rot90(grid, k=-1)
    
    elif aug_type == AUG_ROT_180 or aug_type == 'O':
        # Rotate 180 degrees
        return np.rot90(grid, k=2)
    
    elif aug_type == AUG_FLIP_V or aug_type == 'V':
        # Flip vertically (up-down)
        return np.flipud(grid)
    
    elif aug_type == AUG_FLIP_H or aug_type == 'H':
        # Flip horizontally (left-right)
        return np.fliplr(grid)
    
    elif aug_type == AUG_TRANSPOSE or aug_type == 'T':
        # Transpose
        return grid.T
    
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def apply_augmentation_pattern(
    grid: np.ndarray, 
    pattern: str,
    grid_index: Optional[int] = None
) -> np.ndarray:
    """
    Apply augmentation pattern to grid.
    
    For grid-based augmentation (MIT SEAL style):
    - Pattern is a string where each position corresponds to a grid cell
    - But for simplicity, we apply ONE augmentation to the entire grid
    - Use grid_index to select which augmentation from pattern
    
    Args:
        grid: 2D numpy array (H, W)
        pattern: String of augmentation codes (e.g., "L.R......")
        grid_index: Index to select from pattern (if None, use first)
    
    Returns:
        Augmented grid
    """
    if grid_index is None:
        grid_index = 0
    
    # Ensure index is valid
    if grid_index >= len(pattern):
        aug_type = AUG_NONE
    else:
        aug_type = pattern[grid_index]
    
    return apply_single_augmentation(grid, aug_type)


def augment_arc_example(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    aug_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment ARC input-output pair with same transformation.
    
    Args:
        input_grid: Input grid (H, W)
        output_grid: Output grid (H', W')
        aug_type: Augmentation type
    
    Returns:
        aug_input, aug_output: Augmented grids
    """
    aug_input = apply_single_augmentation(input_grid, aug_type)
    aug_output = apply_single_augmentation(output_grid, aug_type)
    return aug_input, aug_output


def augment_sudoku(sudoku_grid: np.ndarray, aug_type: str) -> np.ndarray:
    """
    Augment Sudoku grid.
    
    Note: Not all augmentations preserve Sudoku validity.
    Only use: rotation 90, rotation 180, rotation 270, transpose
    These preserve the constraint structure.
    
    Args:
        sudoku_grid: 9x9 Sudoku grid
        aug_type: Augmentation type
    
    Returns:
        Augmented Sudoku grid
    """
    # For Sudoku, only allow rotations and transpose
    valid_augs = [AUG_NONE, AUG_ROT_LEFT, AUG_ROT_RIGHT, AUG_ROT_180, AUG_TRANSPOSE]
    
    if aug_type not in valid_augs:
        # Fall back to no augmentation
        aug_type = AUG_NONE
    
    return apply_single_augmentation(sudoku_grid, aug_type)


def apply_augmentation_sequence_to_whole_grid(
    grid: np.ndarray,
    aug_pattern: str
) -> np.ndarray:
    """
    Apply augmentation sequence to the whole grid sequentially.
    
    The pattern is a string where each character represents an augmentation
    to apply to the entire grid in sequence. For example:
    - "RL......" means: apply R, then L (which reverts back to original)
    - "L.R....." means: apply L, then no change, then R
    
    Args:
        grid: 2D numpy array (H, W) - typically 9x9 Sudoku
        aug_pattern: String of augmentation codes (e.g., "RL......")
                    Each position can be '.' (no change), 'L' (left), or 'R' (right)
    
    Returns:
        Augmented grid (same shape as input)
    """
    result = grid.copy()
    
    # NOTE (NR): Original implementation applied all non-'.' augmentations
    # sequentially to the SAME grid, effectively composing them:
    #   result = a_k(... a_2(a_1(grid)) ...)
    #
    # For the TinyRecursiveModels meta-training setup, we keep this helper
    # unchanged (it is still used in some places), but the *batch-level*
    # augmentation logic has been updated in apply_augmentation_patterns_to_batch
    # to optionally generate multiple augmented examples per pattern instead of
    # composing all active ops into a single grid.
    #
    # Here we preserve the original behaviour for callers that still expect a
    # single composed augmentation.
    for aug_char in aug_pattern:
        if aug_char in ALL_PATTERN_VALUES:
            result = apply_single_augmentation(result, aug_char)
        # Ignore invalid characters
    
    return result


def pattern_from_3x2_grid(aug_grid: np.ndarray) -> str:
    """
    Convert a 3x2 augmentation grid to a pattern string.
    
    The 3x2 grid has position-specific binary choices:
    - Position (0,0): '.' or 'L'
    - Position (0,1): '.' or 'R'
    - Position (1,0): '.' or 'H'
    - Position (1,1): '.' or 'V'
    - Position (2,0): '.' or 'O'
    - Position (2,1): '.' or 'T'
    
    The pattern is read column by column: first all positions from column 0,
    then all positions from column 1.
    For example:
    [[L, R], [H, V], [O, T]] -> "LHORVT" (first column: L, H, O then second column: R, V, T)
    
    Args:
        aug_grid: 3x2 numpy array of augmentation characters
                 Each position must be one of its allowed two choices
    
    Returns:
        Pattern string of length 6
    """
    pattern = []
    # First column: 3 positions
    for i in range(3):
        char = aug_grid[i, 0]
        # Validate: must be one of the allowed choices for this position
        allowed = POSITION_TO_AUG_CHOICES.get((i, 0), ['.'])
        if char not in allowed:
            char = '.'  # Default to no change if invalid
        pattern.append(char)
    # Second column: 3 positions
    for i in range(3):
        char = aug_grid[i, 1]
        # Validate: must be one of the allowed choices for this position
        allowed = POSITION_TO_AUG_CHOICES.get((i, 1), ['.'])
        if char not in allowed:
            char = '.'  # Default to no change if invalid
        pattern.append(char)
    return ''.join(pattern)


def inverse_augmentation(aug_type: str) -> str:
    """
    Get the inverse augmentation (to undo an augmentation).
    
    Args:
        aug_type: Original augmentation
    
    Returns:
        Inverse augmentation type
    """
    if aug_type == AUG_NONE:
        return AUG_NONE
    elif aug_type == AUG_ROT_LEFT:
        return AUG_ROT_RIGHT
    elif aug_type == AUG_ROT_RIGHT:
        return AUG_ROT_LEFT
    elif aug_type == AUG_ROT_180:
        return AUG_ROT_180  # Self-inverse
    elif aug_type == AUG_FLIP_V:
        return AUG_FLIP_V  # Self-inverse
    elif aug_type == AUG_FLIP_H:
        return AUG_FLIP_H  # Self-inverse
    elif aug_type == AUG_TRANSPOSE:
        return AUG_TRANSPOSE  # Self-inverse
    else:
        return AUG_NONE


class AugmentationSampler:
    """
    Samples augmentation patterns for meta-learning.
    Can be used for:
    1. Random baseline
    2. Meta-model guided sampling
    
    Supports 3x2 grid (6 positions) with position-specific binary choices:
    - Position (0,0): '.' or 'L'
    - Position (0,1): '.' or 'R'
    - Position (1,0): '.' or 'H'
    - Position (1,1): '.' or 'V'
    - Position (2,0): '.' or 'O'
    - Position (2,1): '.' or 'T'
    """
    
    def __init__(self, pattern_length: int = 6, temperature: float = 1.0):
        """
        Args:
            pattern_length: Length of augmentation pattern string (6 for 3x2 grid)
            temperature: Sampling temperature (higher = more random)
        """
        self.pattern_length = pattern_length
        self.temperature = temperature
        self.num_aug_types = 2  # Binary: each position has 2 choices
    
    def sample_random_pattern(self) -> str:
        """Sample random augmentation pattern."""
        pattern = []
        for pos_idx in range(6):
            # Random binary choice for this position
            choice = np.random.randint(0, 2)
            char = POSITION_INDEX_TO_CHOICES[pos_idx][choice]
            pattern.append(char)
        return ''.join(pattern)
    
    def sample_from_logits(self, logits: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """
        Sample augmentation pattern from meta-model logits.
        
        Args:
            logits: Tensor of shape (6, 2) for 3x2 grid with binary choices per position
        
        Returns:
            pattern: Sampled augmentation pattern string
            log_probs: Log probabilities of sampled actions (for REINFORCE)
        """
        # Apply temperature
        logits = logits / self.temperature
        
        # Binary sampling per position
        probs = torch.softmax(logits, dim=-1)  # [6, 2]
        sampled_indices = torch.multinomial(probs.view(-1, 2), num_samples=1).view(-1)  # [6]
        
        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)  # [6, 2]
        sampled_log_probs = log_probs.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)  # [6]
        
        # Convert to pattern string using position-specific mappings
        pattern = []
        for pos_idx, idx in enumerate(sampled_indices):
            char = POSITION_INDEX_TO_CHOICES[pos_idx][idx.item()]
            pattern.append(char)
        
        return ''.join(pattern), sampled_log_probs.sum()
    
    def greedy_from_logits(self, logits: torch.Tensor) -> str:
        """
        Greedy selection from logits (for evaluation).
        
        Args:
            logits: Tensor of shape (6, 2) for 3x2 grid
        
        Returns:
            pattern: Greedy augmentation pattern string
        """
        indices = torch.argmax(logits, dim=-1)  # [6]
        pattern = []
        for pos_idx, idx in enumerate(indices):
            char = POSITION_INDEX_TO_CHOICES[pos_idx][idx.item()]
            pattern.append(char)
        return ''.join(pattern)


def flatten_grid_to_sequence(grid: np.ndarray) -> np.ndarray:
    """
    Flatten 2D grid to 1D sequence (row-major order).
    
    Args:
        grid: 2D array (H, W)
    
    Returns:
        Flattened array (H*W,)
    """
    return grid.flatten()


def unflatten_sequence_to_grid(sequence: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Unflatten 1D sequence to 2D grid.
    
    Args:
        sequence: 1D array (H*W,)
        height: Grid height
        width: Grid width
    
    Returns:
        2D grid (H, W)
    """
    return sequence.reshape(height, width)


# Utility for batch augmentation
def augment_batch(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    aug_pattern: str,
    grid_height: int,
    grid_width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply augmentation to a batch of flattened grids.
    
    Args:
        inputs: Batch of input sequences (B, L) where L = H*W
        labels: Batch of label sequences (B, L)
        aug_pattern: Augmentation pattern string
        grid_height: Height of grid
        grid_width: Width of grid
    
    Returns:
        aug_inputs, aug_labels: Augmented tensors
    """
    batch_size = inputs.shape[0]
    
    # Convert to numpy for augmentation
    inputs_np = inputs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    aug_inputs_list = []
    aug_labels_list = []
    
    for i in range(batch_size):
        # Unflatten
        input_grid = unflatten_sequence_to_grid(inputs_np[i], grid_height, grid_width)
        label_grid = unflatten_sequence_to_grid(labels_np[i], grid_height, grid_width)
        
        # Select augmentation for this example
        aug_type = aug_pattern[i % len(aug_pattern)]
        
        # Augment
        aug_input_grid = apply_single_augmentation(input_grid, aug_type)
        aug_label_grid = apply_single_augmentation(label_grid, aug_type)
        
        # Flatten back
        aug_inputs_list.append(flatten_grid_to_sequence(aug_input_grid))
        aug_labels_list.append(flatten_grid_to_sequence(aug_label_grid))
    
    # Convert back to tensors
    aug_inputs = torch.tensor(np.stack(aug_inputs_list), dtype=inputs.dtype, device=inputs.device)
    aug_labels = torch.tensor(np.stack(aug_labels_list), dtype=labels.dtype, device=labels.device)
    
    return aug_inputs, aug_labels

