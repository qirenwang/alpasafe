from __future__ import annotations

from safeworld.data.schema import AlpamayoProposal, TopKAlpamayoProposal


def parse_proposal(row: dict[str, object]) -> AlpamayoProposal:
    return AlpamayoProposal.from_dict(row)


def parse_topk_proposal(row: dict[str, object]) -> TopKAlpamayoProposal:
    return TopKAlpamayoProposal.from_dict(row)
