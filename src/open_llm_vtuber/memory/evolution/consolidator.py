"""Consolidator — merges similar semantic memories and decays cold ones.

Operations:
1. Merge: Combine knowledge entries with high content similarity
2. Decay: Reduce importance of entries not accessed recently
3. Promote: Increase importance of frequently accessed entries
"""

from typing import Dict, List, Any
from loguru import logger

from ..layers.semantic import SemanticLayer


class Consolidator:
    """Consolidates semantic memory by merging similar entries and decaying cold ones."""

    def __init__(self, semantic: SemanticLayer):
        self._semantic = semantic

    async def consolidate(self) -> Dict[str, int]:
        """Run consolidation cycle.

        Returns:
            Stats dict with counts of merged, decayed, promoted entries.
        """
        stats = {"merged": 0, "decayed": 0, "promoted": 0}

        all_knowledge = await self._semantic.list_all(limit=500)
        if not all_knowledge:
            return stats

        # ── 1. Merge similar entries ──────────────────────────
        merged_ids = set()
        for i, entry_a in enumerate(all_knowledge):
            if entry_a.get("id") in merged_ids:
                continue
            for j, entry_b in enumerate(all_knowledge):
                if j <= i or entry_b.get("id") in merged_ids:
                    continue

                # Simple similarity: check if content overlap is significant
                if self._should_merge(entry_a, entry_b):
                    merged_content = self._merge_contents(entry_a, entry_b)
                    await self._semantic.update_knowledge(
                        entry_a["id"],
                        {
                            **entry_a,
                            "content": merged_content,
                            "importance": max(
                                entry_a.get("importance", 0.5),
                                entry_b.get("importance", 0.5),
                            ),
                        },
                    )
                    await self._semantic.delete_knowledge(entry_b["id"])
                    merged_ids.add(entry_b["id"])
                    stats["merged"] += 1

        # ── 2. Decay cold entries ─────────────────────────────
        for entry in all_knowledge:
            if entry.get("id") in merged_ids:
                continue
            importance = entry.get("importance", 0.5)
            access_count = entry.get("access_count", 0)

            if access_count == 0 and importance > 0.2:
                new_importance = max(0.1, importance - 0.1)
                await self._semantic.update_knowledge(
                    entry["id"],
                    {**entry, "importance": new_importance},
                )
                stats["decayed"] += 1

        # ── 3. Promote hot entries ────────────────────────────
        for entry in all_knowledge:
            if entry.get("id") in merged_ids:
                continue
            access_count = entry.get("access_count", 0)
            if access_count >= 3:
                importance = entry.get("importance", 0.5)
                new_importance = min(1.0, importance + 0.05)
                await self._semantic.update_knowledge(
                    entry["id"],
                    {**entry, "importance": new_importance},
                )
                stats["promoted"] += 1

        logger.info(
            f"Consolidator: merged={stats['merged']}, "
            f"decayed={stats['decayed']}, promoted={stats['promoted']}"
        )
        return stats

    def _should_merge(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """Check if two knowledge entries should be merged."""
        content_a = a.get("content", "").lower()
        content_b = b.get("content", "").lower()

        # Simple heuristic: if contents share significant words
        words_a = set(content_a.split())
        words_b = set(content_b.split())
        if not words_a or not words_b:
            return False

        overlap = len(words_a & words_b)
        min_len = min(len(words_a), len(words_b))
        jaccard = overlap / min_len if min_len > 0 else 0

        return jaccard > 0.7  # High overlap threshold

    def _merge_contents(self, a: Dict[str, Any], b: Dict[str, Any]) -> str:
        """Merge two knowledge entries' contents."""
        content_a = a.get("content", "")
        content_b = b.get("content", "")
        # Keep the longer/more detailed content
        return content_a if len(content_a) >= len(content_b) else content_b
