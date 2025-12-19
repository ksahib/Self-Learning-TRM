from typing import Optional
import os
import csv
import json
import numpy as np
import pandas as pd

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    # Default: original Sudoku Extreme dataset. Override to e.g. "LangAGI-Lab/Sudoku-Easy".
    source_repo: str = "sapientinc/sudoku-extreme"

    output_dir: str = "data/sudoku-extreme-full"

    subsample_size: Optional[int] = None
    subsample_size_test: Optional[int] = None
    min_difficulty: Optional[int] = None
    max_difficulty: Optional[int] = None
    num_aug: int = 0


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def convert_subset(set_name: str, config: DataProcessConfig):
    # Read source dataset into lists of boards & solutions.
    inputs = []
    labels = []

    if config.source_repo == "LangAGI-Lab/Sudoku-Easy":
        # LangAGI-Lab/Sudoku-Easy is stored as Parquet with columns:
        # - initial_board: puzzle string ('.' for blanks)
        # - solution: solution string
        # - rows, cols: board dimensions (4x4 in this dataset)
        #
        # We mirror the earlier CSV logic but adapt to variable board size.
        # Try common Parquet filenames for HF datasets.
        parquet_candidates = [
            f"{set_name}.parquet",
            f"{set_name}-00000-of-00001.parquet",
        ]
        parquet_path = None
        for fname in parquet_candidates:
            try:
                parquet_path = hf_hub_download(
                    config.source_repo, fname, repo_type="dataset"
                )
                break
            except Exception:
                parquet_path = None
        if parquet_path is None:
            raise FileNotFoundError(
                f"Could not find a Parquet file for split '{set_name}' "
                f"in repo '{config.source_repo}'. Tried: {parquet_candidates}"
            )

        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            q = row["initial_board"]
            a = row["solution"]
            rows_ = int(row["rows"])
            cols_ = int(row["cols"])

            assert rows_ == cols_, "Only square Sudokus are supported"
            size = rows_
            assert len(q) == size * size and len(a) == size * size

            inputs.append(
                np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8).reshape(
                    size, size
                )
                - ord("0")
            )
            labels.append(
                np.frombuffer(a.encode(), dtype=np.uint8).reshape(size, size)
                - ord("0")
            )
    else:
        # Original CSV-based format from sapientinc/sudoku-extreme (and similar).
        with open(
            hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"),
            newline="",
        ) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for source, q, a, rating in reader:
                rating_int = int(rating)
                # Filter by difficulty range
                if config.min_difficulty is not None and rating_int < config.min_difficulty:
                    continue
                if config.max_difficulty is not None and rating_int > config.max_difficulty:
                    continue
                
                assert len(q) == 81 and len(a) == 81

                inputs.append(
                    np.frombuffer(
                        q.replace(".", "0").encode(), dtype=np.uint8
                    ).reshape(9, 9)
                    - ord("0")
                )
                labels.append(
                    np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9)
                    - ord("0")
                )

    # Subsampling: different logic for train vs test
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]
    elif set_name == "test" and config.subsample_size_test is not None:
        total_samples = len(inputs)
        if config.subsample_size_test < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size_test, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    num_augments = config.num_aug if set_name == "train" else 0

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            # Push puzzle (only single example)
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            
        # Push group
        results["group_indices"].append(puzzle_id)
        
    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        
        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    seq_len = int(results["inputs"].shape[1])
    vocab_size = int(results["inputs"].max()) + 1  # values already offset by +1

    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=vocab_size,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
