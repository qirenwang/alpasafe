"""Candidate-specific reasoning↔trajectory alignment recovery (T24R5, v2.1).

Background. The completed K=2 raw artifact stored, in BOTH candidate records, the
same serialized two-element reasoning array, e.g.::

    "['Keep distance to the lead vehicle since it is stopped ahead'
      'Keep distance to the lead vehicle since it is directly ahead in our lane']"

That blocks reasoning-conditioned training because a record does not, on its own,
say which element belongs to which trajectory.

Why the mapping is recoverable WITHOUT guessing and WITHOUT a rerun.
``alpamayo1_5.models.alpamayo1_5.Alpamayo1_5.sample_trajectories_from_data_with_
vlm_rollout`` rearranges BOTH the trajectories and the chain-of-thought text
tokens with the *same* layout ``[B, num_traj_sets, num_traj_samples]`` (the source
comment: "rearrange text tokens to shape [B, ns, nj] to match trajectory shape").
Both come from the same ``vlm_outputs.sequences`` batch
(``num_return_sequences = num_traj_samples``), so element ``j`` of the reasoning
array corresponds to trajectory sample ``j``. The driver dump-hook bug indexed the
``num_traj_sets`` axis (size 1) instead of ``num_traj_samples``, so every record
received the full ``[nj]`` array — but the array's ELEMENT ORDER is exactly the
sample order. Therefore ``reasoning_array[i]`` ↔ ``candidate sample_index i``.

This module parses the preserved array, splits it deterministically by sample
index, and provides :func:`assert_unambiguous_reasoning` so the real training CLI
rejects any record whose reasoning is still a serialized multi-element array.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

__all__ = [
    "ReasoningAmbiguityError",
    "parse_serialized_reasoning_array",
    "looks_like_serialized_array",
    "split_reasoning_by_sample_index",
    "assert_unambiguous_reasoning",
    "CandidateReasoning",
    "recover_candidate_reasoning",
]

# Matches one Python/numpy quoted string literal ('...' or "..."), honoring escapes.
_QUOTED = re.compile(r"""'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*\"""", re.DOTALL)


class ReasoningAmbiguityError(ValueError):
    """Raised when a reasoning trace is not a single unambiguous string."""


def parse_serialized_reasoning_array(text: str) -> list[str]:
    """Parse a ``str(np.ndarray[str])`` / ``repr(list[str])`` into its elements.

    numpy prints string arrays as ``['a' 'b']`` (whitespace-separated, each element
    repr-quoted), which is not valid Python literal syntax, so we extract the
    top-level quoted tokens and unescape each with :func:`ast.literal_eval`.
    """
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"not a serialized array: {text!r}")
    inner = s[1:-1]
    tokens = _QUOTED.findall(inner)
    if not tokens:
        raise ValueError(f"no string elements found in: {text!r}")
    out: list[str] = []
    for tok in tokens:
        try:
            out.append(str(ast.literal_eval(tok)))
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"could not parse element {tok!r}: {exc}") from exc
    return out


def looks_like_serialized_array(text: str | None) -> bool:
    """True if ``text`` looks like a serialized array of >= 1 string element."""
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return False
    return bool(_QUOTED.findall(s[1:-1]))


def split_reasoning_by_sample_index(text: str, num_candidates: int) -> list[str]:
    """Split a preserved reasoning array into ``num_candidates`` per-sample strings.

    Element order IS sample order (see module docstring). Raises if the array does
    not have exactly ``num_candidates`` elements — we never pad, truncate, or guess.
    """
    elems = parse_serialized_reasoning_array(text)
    if len(elems) != num_candidates:
        raise ReasoningAmbiguityError(
            f"reasoning array has {len(elems)} elements but {num_candidates} candidates; "
            "cannot align without guessing"
        )
    return elems


def assert_unambiguous_reasoning(reasoning_trace: str | None, *, context: str = "real CLI") -> None:
    """Guard for the real training/eval CLI: reject ambiguous reasoning records.

    A valid record carries exactly ONE reasoning string for ONE candidate. A trace
    that is still a serialized multi-element array (the un-split form) is rejected.
    """
    if reasoning_trace is None or not str(reasoning_trace).strip():
        raise ReasoningAmbiguityError(f"{context}: empty reasoning trace")
    if looks_like_serialized_array(reasoning_trace):
        elems = parse_serialized_reasoning_array(reasoning_trace)
        if len(elems) != 1:
            raise ReasoningAmbiguityError(
                f"{context}: reasoning trace is a serialized {len(elems)}-element array "
                f"(ambiguous candidate→reasoning mapping); split by sample index first"
            )


@dataclass(frozen=True)
class CandidateReasoning:
    """One candidate's recovered, unambiguous reasoning."""

    candidate_index: int
    reasoning_trace: str
    recovery: str  # provenance note for how this was obtained


def recover_candidate_reasoning(
    records: list[dict],
    *,
    reasoning_key: str = "reasoning_text",
    index_key: str = "candidate_index",
) -> list[CandidateReasoning]:
    """Recover one unambiguous reasoning per candidate from raw K-sample records.

    Each record must carry the preserved reasoning array (identical across records)
    and a candidate/sample index. The array is split by sample index per the proven
    model contract. Raises if records disagree on the array or indices are not a
    contiguous ``0..N-1`` permutation.
    """
    if not records:
        raise ValueError("no records")
    arrays = {r.get(reasoning_key) for r in records}
    if len(arrays) != 1:
        raise ReasoningAmbiguityError(
            "records carry different reasoning arrays; expected one shared preserved array"
        )
    shared = next(iter(arrays))
    n = len(records)
    elems = split_reasoning_by_sample_index(shared, n)
    indices = sorted(int(r[index_key]) for r in records)
    if indices != list(range(n)):
        raise ReasoningAmbiguityError(f"candidate indices must be 0..{n - 1}, got {indices}")
    out: list[CandidateReasoning] = []
    for r in records:
        i = int(r[index_key])
        out.append(
            CandidateReasoning(
                candidate_index=i,
                reasoning_trace=elems[i],
                recovery=(
                    "split preserved CoT array by sample index per Alpamayo1_5 "
                    "[B,ns,nj] reshape contract (nj==sample order)"
                ),
            )
        )
    out.sort(key=lambda c: c.candidate_index)
    return out
