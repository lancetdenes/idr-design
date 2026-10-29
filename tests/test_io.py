"""Tests for idrdesign.io — FASTA read/write."""
from pathlib import Path
import pytest

from idrdesign.io import read_fasta, write_fasta, select_record


def test_read_fasta_single_record(tmp_path: Path):
    p = tmp_path / "one.fasta"
    p.write_text(">foo\nACDEFG\n")
    records = read_fasta(p)
    assert records == {"foo": "ACDEFG"}


def test_read_fasta_multi_record_with_wrapped_lines(tmp_path: Path):
    p = tmp_path / "multi.fasta"
    p.write_text(">a\nACDE\nFGHI\n>b\nKLMN\n")
    records = read_fasta(p)
    assert records == {"a": "ACDEFGHI", "b": "KLMN"}


def test_read_fasta_strips_whitespace_and_ignores_blank_lines(tmp_path: Path):
    p = tmp_path / "ws.fasta"
    p.write_text(">a\n  ACDE  \n\n   \nFGHI\n")
    records = read_fasta(p)
    assert records == {"a": "ACDEFGHI"}


def test_read_fasta_header_keeps_only_first_token(tmp_path: Path):
    p = tmp_path / "header.fasta"
    p.write_text(">foo some description here\nACDE\n")
    records = read_fasta(p)
    assert records == {"foo": "ACDE"}


def test_read_fasta_raises_on_duplicate_headers(tmp_path: Path):
    p = tmp_path / "dup.fasta"
    p.write_text(">foo\nACDE\n>foo\nKLMN\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_fasta(p)


def test_select_record_found():
    records = {"WT": "ACDE", "Mut": "FGHI"}
    assert select_record(records, "WT") == "ACDE"


def test_select_record_missing_lists_available():
    records = {"WT": "ACDE", "Mut": "FGHI"}
    with pytest.raises(KeyError, match="WT.*Mut"):
        select_record(records, "Missing")


def test_write_fasta_roundtrip(tmp_path: Path):
    p = tmp_path / "out.fasta"
    records = {"a": "ACDE", "b": "FGHI"}
    write_fasta(p, records, line_width=60)
    assert read_fasta(p) == records


def test_write_fasta_wraps_long_sequences(tmp_path: Path):
    p = tmp_path / "out.fasta"
    write_fasta(p, {"a": "A" * 125}, line_width=60)
    lines = p.read_text().splitlines()
    # header + 3 lines of sequence (60, 60, 5)
    assert lines[0] == ">a"
    assert lines[1] == "A" * 60
    assert lines[2] == "A" * 60
    assert lines[3] == "A" * 5
