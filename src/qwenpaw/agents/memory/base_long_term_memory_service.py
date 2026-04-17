"""Abstract base class for long term memory."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from agentscope.message import Msg
from agentscope.tool import ToolResponse

from ..utils import Registry
from ...config import load_agent_config

logger = logging.getLogger(__name__)


class BaseLongTermMemoryService(ABC):
    """Abstract base class defining the long term memory interface."""

    def __init__(self, agent_id: str, working_dir: str | Path):
        self.agent_id: str = agent_id
        self.working_dir: Path = Path(working_dir)

    @abstractmethod
    async def start(self) -> None:
        """Start the long term memory lifecycle."""

    @abstractmethod
    async def close(self) -> bool:
        """Close the long term memory and perform cleanup."""

    @property
    async def memory_prompt(self) -> str:
        """"""

    @abstractmethod
    def list_tools(self) -> list[Callable[..., ToolResponse]]:
        """"""

    @abstractmethod
    async def summarize(
            self,
            msgs: list[Msg],
            **kwargs,
    ) -> Any:
        """A developer-designed method to record information from the given
        input message(s) to the long-term memory."""

    async def retrieve(
            self,
            msg: Msg | list[Msg],
            **kwargs,
    ) -> list[Msg]:
        """A developer-designed method to retrieve information from the
        long-term memory based on the given input message(s). The retrieved
        information will be added to the system prompt of the agent."""

    @abstractmethod
    async def dream(self, **kwargs) -> Any:
        """Run one dream-based memory optimization task."""

    @abstractmethod
    async def proactive(self, **kwargs) -> Any:
        """Run one proactive memory optimization task."""

    def add_summarize_task(self, messages: list[Msg], **kwargs) -> None:
        """Add an asynchronous summary task for the given messages."""

    def list_summarize_status(self) -> list[dict]:
        """List the status of all summarize tasks."""


ltms_registry = Registry[BaseLongTermMemoryService]()


def get_long_term_memory_service(agent_id: str, working_dir: str | Path) -> BaseLongTermMemoryService:
    agent_config = load_agent_config(agent_id)
    backend_name: str = agent_config.running.long_term_memory_backend
    impl_class = ltms_registry.get(backend_name)

    if impl_class is None:
        raise RuntimeError(
            f"Long term memory backend '{backend_name}' not registered. "
            f"Available: {ltms_registry.list_registered()}",
        )
    return impl_class(agent_id=agent_id, working_dir=working_dir)
