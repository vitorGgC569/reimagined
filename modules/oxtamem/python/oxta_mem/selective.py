# -*- coding: utf-8 -*-
"""Selective-write policy for OxtaMem ("Camada 1").

Instead of storing every item, gate writes by *surprise*: an item is worth
remembering only if it is novel relative to what is already stored.  Novelty is
measured as the cosine distance from the item's embedding to its nearest
existing memory (via ``search_similar_scored``).  Redundant near-duplicates are
skipped, so a fixed write budget is spent on a diverse, high-coverage set rather
than on many copies of the same thing.

This is the intra-memory ("did I already know this?") flavour of surprise and
needs no model signal.  An external surprise/learning-progress score (e.g. the
model's prediction loss on the item) can be supplied via ``extra_surprise`` to
combine model-driven novelty with memory-driven novelty.
"""

from __future__ import annotations
from typing import Optional, Sequence


class SelectiveWriter:
    """Novelty-gated writer around a ``PyGeodesicEngine``-like store.

    Parameters
    ----------
    engine:
        Any object exposing ``search_similar_scored(vector, k) -> [(distance, value)]``
        and ``write_with_vector(token_id, value, vector) -> int``.
    novelty_threshold:
        Minimum cosine distance (0 = identical, larger = more different) to the
        nearest existing memory for a write to be accepted.  Candidates closer
        than this are treated as redundant and skipped.  Higher = stricter =
        fewer, more diverse writes.
    surprise_floor:
        If an external surprise score is given to :meth:`offer`, a candidate is
        written whenever that score is at least this floor, *regardless* of
        novelty — so genuinely surprising items are never dropped as duplicates.
    """

    def __init__(
        self,
        engine,
        novelty_threshold: float = 0.30,
        surprise_floor: float = 1.0,
    ) -> None:
        if not (0.0 <= novelty_threshold <= 2.0):
            raise ValueError("novelty_threshold must be in [0, 2] (cosine distance)")
        self.engine = engine
        self.novelty_threshold = float(novelty_threshold)
        self.surprise_floor = float(surprise_floor)
        self.seen = 0
        self.written = 0
        self.skipped = 0

    def _nearest_distance(self, vector: Sequence[float]) -> Optional[float]:
        hits = self.engine.search_similar_scored(list(vector), 1)
        if not hits:
            return None
        return float(hits[0][0])

    def offer(
        self,
        token_id: str,
        value: bytes,
        vector: Sequence[float],
        extra_surprise: Optional[float] = None,
    ) -> bool:
        """Consider writing (token_id, value, vector). Returns True if written.

        Written iff the store is empty, OR the nearest memory is far enough
        (distance >= novelty_threshold), OR an external ``extra_surprise`` score
        clears ``surprise_floor``.
        """
        self.seen += 1

        if extra_surprise is not None and extra_surprise >= self.surprise_floor:
            self.engine.write_with_vector(token_id, value, list(vector))
            self.written += 1
            return True

        nearest = self._nearest_distance(vector)
        if nearest is not None and nearest < self.novelty_threshold:
            self.skipped += 1
            return False

        self.engine.write_with_vector(token_id, value, list(vector))
        self.written += 1
        return True

    def stats(self) -> dict:
        return {
            "seen": self.seen,
            "written": self.written,
            "skipped": self.skipped,
            "write_rate": (self.written / self.seen) if self.seen else 0.0,
        }
