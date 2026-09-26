"""Dataset loading."""

from pathlib import Path
import pandas as pd
from .preprocessing import clean_sequence

def _find_column(columns, wanted: str) -> str:
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    key = wanted.strip().lower()
    if key not in lookup:
        raise ValueError(
            f"Required column '{wanted}' not found. Available: {list(columns)}"
        )
    return lookup[key]

def load_sequence_file(
    path,
    sequence_column: str = "sequence",
    target_column: str = "class",
    species: str | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, sep=None, engine="python")
    seq_col = _find_column(frame.columns, sequence_column)
    target_col = _find_column(frame.columns, target_column)

    rows = []
    for _, row in frame[[seq_col, target_col]].dropna().iterrows():
        try:
            seq = clean_sequence(row[seq_col])
        except ValueError:
            continue
        item = {"sequence": seq, "class": row[target_col]}
        if species is not None:
            item["species"] = species
        rows.append(item)

    return pd.DataFrame(rows)

def load_many(paths, sequence_column="sequence", target_column="class"):
    frames = []
    for path in paths:
        p = Path(path)
        frames.append(
            load_sequence_file(
                p,
                sequence_column=sequence_column,
                target_column=target_column,
                species=p.stem,
            )
        )
    if not frames:
        raise ValueError("At least one dataset is required.")
    return pd.concat(frames, ignore_index=True)
