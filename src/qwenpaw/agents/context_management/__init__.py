"""Context management package."""

from pathlib import Path

from .base_context_management import (
    BaseContextManagement,
    cm_registry,
    get_context_management,
)
from .reme_in_memory_memory import ReMeInMemoryMemory

__all__ = [
    "BaseContextManagement",
    "cm_registry",
    "get_context_management",
    "ReMeInMemoryMemory",
    "create_in_memory_memory",
]


def create_in_memory_memory(
    agent_id: str,
    working_dir: str | Path,
) -> ReMeInMemoryMemory:
    """Create a ReMeInMemoryMemory instance for the given agent.

    Args:
        agent_id: Agent ID for config loading.
        working_dir: Working directory for dialog storage.

    Returns:
        ReMeInMemoryMemory instance.
    """
    from ..utils import get_token_counter
    from ...config import load_agent_config

    agent_config = load_agent_config(agent_id)
    token_counter = get_token_counter(agent_config)
    dialog_path = Path(working_dir) / "dialog"

    return ReMeInMemoryMemory(
        token_counter=token_counter,
        dialog_path=dialog_path,
    )