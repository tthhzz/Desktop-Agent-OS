"""5-layer memory system for autonomous desktop pet agent.

Layers:
- Sensory: Short-lived perceptual buffer (camera/screen frames)
- Working: Current conversation context
- Episodic: Conversation events with emotional annotations
- Semantic: Long-term knowledge (facts, preferences, concepts)
- Skill: Reusable tool-chain patterns
"""

from .memory_system import MemorySystem

__all__ = ["MemorySystem"]
