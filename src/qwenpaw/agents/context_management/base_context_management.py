"""Base context management and factory module."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils import Registry
from ...config import load_agent_config

if TYPE_CHECKING:
    from agentscope.agent import ReActAgent
    from agentscope.message import Msg
    from ..memory import BaseLongTermMemoryService

logger = logging.getLogger(__name__)


class BaseContextManagement:
    """Base class for context management implementations."""

    def __init__(
        self,
        agent_id: str,
        working_dir: str | Path,
        memory_manager: "BaseLongTermMemoryService | None" = None,
    ):
        self.agent_id: str = agent_id
        self.working_dir: Path = Path(working_dir)
        self.memory_manager: "BaseLongTermMemoryService | None" = memory_manager

    def pre_reply(
        self,
        agent: "ReActAgent",
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Pre-reply hook. Args match agentscope hook signature."""
        ...

    def post_reply(
        self,
        agent: "ReActAgent",
        kwargs: dict[str, Any],
        output: "Msg",
    ) -> "Msg | None":
        """Post-reply hook. Args match agentscope hook signature."""
        ...

    def pre_reasoning(
        self,
        agent: "ReActAgent",
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Pre-reasoning hook. Args match agentscope hook signature."""
        ...

    def post_acting(
        self,
        agent: "ReActAgent",
        kwargs: dict[str, Any],
        output: Any,
    ) -> "Msg | None":
        """Post-acting hook. Args match agentscope hook signature."""
        ...


cm_registry = Registry[BaseContextManagement]()


def get_context_management(
    agent_id: str,
    working_dir: str | Path,
    memory_manager: "BaseLongTermMemoryService | None" = None,
) -> BaseContextManagement:
    """Create a context management instance for the given agent."""
    agent_config = load_agent_config(agent_id)
    backend_name: str = agent_config.running.context_management_backend
    impl_class = cm_registry.get(backend_name)

    if impl_class is None:
        raise RuntimeError(
            f"Context management backend '{backend_name}' not registered. "
            f"Available: {cm_registry.list_registered()}",
        )
    return impl_class(
        agent_id=agent_id,
        working_dir=working_dir,
        memory_manager=memory_manager,
    )
