# -*- coding: utf-8 -*-
"""Abstract base class for memory managers."""
import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

from agentscope.message import Msg
from agentscope.tool import ToolResponse

from ..utils.registry import Registry
from ...config.config import load_agent_config

logger = logging.getLogger(__name__)


class BaseMemoryManager(ABC):
    """Abstract base class for memory manager backends.

    Lifecycle:
        1. Instantiate with ``working_dir`` and ``agent_id``.
        2. ``await start()`` – initialize storage backend.
        3. Use ``summarize()``, ``memory_search()``, etc. during session.
        4. ``await close()`` – flush and release resources.

    Attributes:
        working_dir: Root directory for persisting memory files.
        agent_id: Unique identifier of the owning agent.
        summary_tasks: Active background summarization tasks; pruned on
            each call to ``add_summarize_task()``.
    """

    def __init__(self, working_dir: str, agent_id: str):
        self.working_dir: str = working_dir
        self.agent_id: str = agent_id
        self.summary_tasks: list[asyncio.Task] = []

    @abstractmethod
    async def start(self) -> None:
        """Initialize the storage backend. Called once after instantiation."""

    @abstractmethod
    async def close(self) -> bool:
        """Flush pending state and release resources.

        Returns:
            ``True`` if shutdown completed cleanly.
        """

    @abstractmethod
    def get_memory_prompt(self, language: str = "zh") -> str:
        """Return the memory guidance prompt for inclusion in the system prompt.

        Args:
            language: Language code (``"zh"`` or ``"en"``).

        Returns:
            Formatted memory guidance string.
        """

    @abstractmethod
    def list_memory_tools(self) -> list[Callable[..., ToolResponse]]:
        """Return tool functions exposed to the agent for memory access.

        Each returned callable may have any signature but must return a
        ``ToolResponse``.  Implementations register whatever memory-related
        tools make sense for the backend (e.g. semantic search, listing).

        Returns:
            Ordered list of tool functions to register with the agent toolkit.
        """

    async def summarize(self, messages: list[Msg], **kwargs) -> str:
        """Summarize conversation messages and persist to memory.

        Args:
            messages: Ordered conversation messages to summarize.
            **kwargs: Implementation-specific options.

        Returns:
            The generated summary string.
        """

    async def retrieve(self, messages: list[Msg] | Msg, **kwargs) -> list[Msg]:
        """Retrieve relevant memory based on the given messages.

        Args:
            messages: One or more conversation messages used as the query.
            **kwargs: Implementation-specific options.

        Returns:
            Retrieved memory messages.
        """

    @abstractmethod
    async def dream(self, **kwargs) -> None:
        """Optimize memory files via a background agent pass.

        Runs a lightweight ReAct agent with file-editing tools to
        consolidate redundant or outdated memory entries.
        """

    def add_summarize_task(self, messages: list[Msg], **kwargs):
        """Schedule a background summarization task without blocking.

        Prunes completed tasks (logging cancellations/failures/results),
        then wraps ``summarize()`` in a new ``asyncio.Task``.

        Args:
            messages: Messages to pass to ``summarize()``.
            **kwargs: Forwarded to ``summarize()``.
        """
        remaining_tasks = []
        for task in self.summary_tasks:
            if task.done():
                if task.cancelled():
                    logger.warning("Summary task was cancelled.")
                    continue
                exc = task.exception()
                if exc is not None:
                    logger.error(f"Summary task failed: {exc}")
                else:
                    logger.info(f"Summary task completed: {task.result()}")
            else:
                remaining_tasks.append(task)
        self.summary_tasks = remaining_tasks

        task = asyncio.create_task(self.summarize(messages=messages, **kwargs))
        self.summary_tasks.append(task)

    async def await_summary_tasks(self) -> str:
        """Wait for all background summary tasks and collect results.

        Returns:
            Concatenated status messages for each task.
        """
        result = ""
        for task in self.summary_tasks:
            if task.done():
                if task.cancelled():
                    logger.warning("Summary task was cancelled.")
                    result += "Summary task was cancelled.\n"
                else:
                    exc = task.exception()
                    if exc is not None:
                        logger.error(f"Summary task failed: {exc}")
                        result += f"Summary task failed: {exc}\n"
                    else:
                        task_result = task.result()
                        logger.info(f"Summary task completed: {task_result}")
                        result += f"Summary task completed: {task_result}\n"
            else:
                try:
                    task_result = await task
                    logger.info(f"Summary task completed: {task_result}")
                    result += f"Summary task completed: {task_result}\n"
                except asyncio.CancelledError:
                    logger.warning("Summary task was cancelled while waiting.")
                    result += "Summary task was cancelled.\n"
                except Exception as e:
                    logger.exception(f"Summary task failed: {e}")
                    result += f"Summary task failed: {e}\n"

        self.summary_tasks.clear()
        return result


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

memory_registry: Registry[BaseMemoryManager] = Registry()


def get_memory_manager(working_dir: str, agent_id: str) -> BaseMemoryManager:
    """Create a memory manager instance for the given agent.

    The backend is resolved from ``agent_config.running.memory_manager_backend``.

    Raises:
        ValueError: When the configured backend has no registered implementation.
    """

    backend = load_agent_config(agent_id).running.memory_manager_backend
    cls = memory_registry.get(backend)
    if cls is None:
        raise ValueError(
            f"Unsupported memory manager backend: '{backend}'. "
            f"Registered: {memory_registry.list_registered()}",
        )
    return cls(working_dir=working_dir, agent_id=agent_id)
