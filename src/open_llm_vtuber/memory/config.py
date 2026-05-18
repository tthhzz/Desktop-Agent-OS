"""Memory system configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class MemoryConfig:
    """Configuration for the 5-layer memory system."""

    # Sensory layer
    sensory_buffer_size: int = 10  # Max recent frames to keep

    # Episodic layer
    episodic_db_path: str = "memory_data/episodic.db"

    # Semantic layer
    semantic_db_path: str = "memory_data/semantic.db"
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_top_k: int = 5

    # Skill layer
    skill_db_path: str = "memory_data/skills.db"
    skill_frequency_threshold: int = 3  # Min occurrences before auto-generating skill

    # Evolution
    reflection_interval: int = 10  # Reflect every N conversation turns
    consolidation_interval: int = 50  # Consolidate every N turns
