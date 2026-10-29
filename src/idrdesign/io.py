"""FASTA read/write for idrdesign.

Simple flat FASTA handling. Rejects duplicate headers (ambiguous selection).
"""
from __future__ import annotations

from pathlib import Path


def read_fasta(path: str | Path) -> dict[str, str]:
    """Read a FASTA file into {name: sequence}.

    Header name is the first whitespace-delimited token after '>'.
    Sequences may be wrapped across lines; whitespace is stripped.
    Raises ValueError on duplicate header names.
    """
    records: dict[str, str] = {}
    name: str | None = None
    seq_parts: list[str] = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(seq_parts)
                header = line[1:].strip()
                new_name = header.split()[0] if header else ""
                if new_name in records:
                    raise ValueError(f"duplicate header in FASTA: {new_name!r}")
                name = new_name
                seq_parts = []
            else:
                seq_parts.append(line.replace(" ", ""))
    if name is not None:
        records[name] = "".join(seq_parts)
    return records


def select_record(records: dict[str, str], name: str) -> str:
    """Return the sequence for `name`, with a helpful error if absent."""
    if name not in records:
        available = ", ".join(records.keys())
        raise KeyError(f"record {name!r} not found; available: {available}")
    return records[name]


def write_fasta(path: str | Path, records: dict[str, str], line_width: int = 60) -> None:
    """Write {name: sequence} to a FASTA file with sequence lines wrapped to line_width."""
    with open(path, "w") as fh:
        for name, seq in records.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), line_width):
                fh.write(seq[i : i + line_width] + "\n")
