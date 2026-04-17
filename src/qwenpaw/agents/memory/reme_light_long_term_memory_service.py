# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches
# mypy: ignore-errors
"""ReMeLight-backed long term memory for agents."""
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit, ToolResponse

from qwenpaw.agents.memory.base_long_term_memory import (
    BaseLongTermMemoryService,
    ltm_registry,
)
from qwenpaw.agents.memory.prompt import get_dream_prompt
from qwenpaw.agents.model_factory import create_model_and_formatter
from qwenpaw.agents.tools import read_file, write_file, edit_file
from qwenpaw.agents.utils import get_token_counter
from qwenpaw.config import load_config
from qwenpaw.config.config import load_agent_config
from qwenpaw.config.context import (
    set_current_workspace_dir,
    set_current_recent_max_bytes,
)
from qwenpaw.constant import EnvVarLoader

if TYPE_CHECKING:
    from reme.memory.file_based.reme_in_memory_memory import ReMeInMemoryMemory

logger = logging.getLogger(__name__)

_EXPECTED_REME_VERSION = "0.3.1.8"
_REME_STORE_VERSION = "v1"


@ltm_registry.register("reme_light")
class ReMeLightLongTermMemoryService(BaseLongTermMemoryService):
    """Long term memory that wraps ReMeLight for agents via composition.

    Holds a ``ReMeLight`` instance (``self._reme``) and delegates all
    lifecycle / search calls to it.

    Capabilities:
    - Memory summarization with file tools via summary_memory()
    - Vector and full-text search via memory_search()
    """

    def __init__(self, working_dir: str, agent_id: str):
        """Initialize with ReMeLight.

        Args:
            working_dir: Working directory for memory storage.
            agent_id: Agent ID for config loading.
        """
        super().__init__(working_dir=working_dir, agent_id=agent_id)
        self._reme_version_ok: bool = self._check_reme_version()
        self._reme = None

        logger.info(
            f"ReMeLightLongTermMemory init: "
            f"agent_id={agent_id}, working_dir={working_dir}",
        )

        backend_env = EnvVarLoader.get_str("MEMORY_STORE_BACKEND", "auto")
        if backend_env == "auto":
            if platform.system() == "Windows":
                memory_backend = "local"
            else:
                try:
                    import chromadb  # noqa: F401 pylint: disable=unused-import

                    memory_backend = "chroma"
                except Exception as e:
                    logger.warning(
                        f"""
chromadb import failed, falling back to `local` backend.
This is often caused by an outdated system SQLite (requires >= 3.35).
Please upgrade your system SQLite to >= 3.35.
See: https://docs.trychroma.com/docs/overview/troubleshooting#sqlite
| Error: {e}
                        """,
                    )
                    memory_backend = "local"
        else:
            memory_backend = backend_env

        from reme.reme_light import ReMeLight

        emb_config = self.get_embedding_config()
        vector_enabled = bool(emb_config["base_url"]) and bool(
            emb_config["model_name"],
        )

        log_cfg = {
            **emb_config,
            "api_key": self._mask_key(emb_config["api_key"]),
        }
        logger.info(
            f"Embedding config: {log_cfg}, vector_enabled={vector_enabled}",
        )

        fts_enabled = EnvVarLoader.get_bool("FTS_ENABLED", True)

        agent_config = load_agent_config(self.agent_id)
        rebuild_on_start = (
            agent_config.running.memory_summary.rebuild_memory_index_on_start
        )

        store_name = "memory"
        effective_rebuild = self._resolve_rebuild_on_start(
            working_dir=working_dir,
            store_version=_REME_STORE_VERSION,
            rebuild_on_start=rebuild_on_start,
        )

        recursive_file_watcher = (
            agent_config.running.memory_summary.recursive_file_watcher
        )

        self._reme = ReMeLight(
            working_dir=working_dir,
            default_embedding_model_config=emb_config,
            default_file_store_config={
                "backend": memory_backend,
                "store_name": store_name,
                "vector_enabled": vector_enabled,
                "fts_enabled": fts_enabled,
            },
            default_file_watcher_config={
                "rebuild_index_on_start": effective_rebuild,
                "recursive": recursive_file_watcher,
            },
        )

        self.summary_toolkit = Toolkit()
        self.summary_toolkit.register_tool_function(read_file)
        self.summary_toolkit.register_tool_function(write_file)
        self.summary_toolkit.register_tool_function(edit_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_key(key: str) -> str:
        """Mask API key, showing first 5 chars only."""
        return key[:5] + "*" * (len(key) - 5) if len(key) > 5 else key

    @staticmethod
    def _resolve_rebuild_on_start(
        working_dir: str,
        store_version: str,
        rebuild_on_start: bool,
    ) -> bool:
        """Return effective rebuild_index_on_start value."""
        sentinel_name = f".reme_store_{store_version}"
        sentinel_path = Path(working_dir) / sentinel_name

        if sentinel_path.exists():
            return rebuild_on_start

        logger.info(
            f"Sentinel '{sentinel_name}' not found, forcing rebuild.",
        )

        try:
            for old in Path(working_dir).glob(".reme_store_*"):
                old.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to remove old sentinels: {e}")

        try:
            sentinel_path.touch()
        except Exception as e:
            logger.warning(f"Failed to create sentinel '{sentinel_name}': {e}")

        return True

    @staticmethod
    def _check_reme_version() -> bool:
        """Return False when installed reme-ai version mismatches."""
        try:
            installed = importlib.metadata.version("reme-ai")
        except importlib.metadata.PackageNotFoundError:
            return True
        if installed != _EXPECTED_REME_VERSION:
            logger.warning(
                f"reme-ai version mismatch: installed={installed}, "
                f"expected={_EXPECTED_REME_VERSION}. "
                f"Run `pip install reme-ai=={_EXPECTED_REME_VERSION}`"
                " to align.",
            )
            return False
        return True

    def _warn_if_version_mismatch(self) -> None:
        """Warn once per call if the cached version check failed."""
        if not self._reme_version_ok:
            logger.warning(
                "reme-ai version mismatch, "
                f"expected={_EXPECTED_REME_VERSION}. "
                f"Run `pip install reme-ai=={_EXPECTED_REME_VERSION}`"
                " to align.",
            )

    def _prepare_model_formatter(self) -> None:
        """Lazily initialize chat_model and formatter if not already set."""
        self._warn_if_version_mismatch()
        if self.chat_model is None or self.formatter is None:
            self.chat_model, self.formatter = create_model_and_formatter(
                self.agent_id,
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_embedding_config(self) -> dict:
        """Return embedding config with priority: config > env var > default."""
        self._warn_if_version_mismatch()
        cfg = load_agent_config(self.agent_id).running.embedding_config
        return {
            "backend": cfg.backend,
            "api_key": cfg.api_key
            or EnvVarLoader.get_str("EMBEDDING_API_KEY"),
            "base_url": cfg.base_url
            or EnvVarLoader.get_str("EMBEDDING_BASE_URL"),
            "model_name": cfg.model_name
            or EnvVarLoader.get_str("EMBEDDING_MODEL_NAME"),
            "dimensions": cfg.dimensions,
            "enable_cache": cfg.enable_cache,
            "use_dimensions": cfg.use_dimensions,
            "max_cache_size": cfg.max_cache_size,
            "max_input_length": cfg.max_input_length,
            "max_batch_size": cfg.max_batch_size,
        }

    async def restart_embedding_model(self):
        """Restart the embedding model with current config."""
        self._warn_if_version_mismatch()
        if self._reme is None:
            return
        await self._reme.restart(
            restart_config={
                "embedding_models": {"default": self.get_embedding_config()},
            },
        )

    # ------------------------------------------------------------------
    # BaseLongTermMemory interface
    # ------------------------------------------------------------------

    async def start(self):
        """Start the ReMeLight lifecycle."""
        self._warn_if_version_mismatch()
        if self._reme is None:
            return None
        return await self._reme.start()

    async def close(self) -> bool:
        """Close ReMeLight and perform cleanup."""
        self._warn_if_version_mismatch()
        logger.info(
            f"ReMeLightLongTermMemory closing: agent_id={self.agent_id}",
        )
        if self._reme is None:
            return True
        result = await self._reme.close()
        logger.info(
            f"ReMeLightLongTermMemory closed: "
            f"agent_id={self.agent_id}, result={result}",
        )
        return result

    async def summary_memory(self, messages: list[Msg], **_kwargs) -> str:
        """Generate a comprehensive summary of the given messages."""
        self._prepare_model_formatter()

        agent_config = load_agent_config(self.agent_id)
        cc = agent_config.running.context_compact

        set_current_workspace_dir(Path(self.working_dir))
        recent_max_bytes = (
            agent_config.running.tool_result_compact.recent_max_bytes
        )
        set_current_recent_max_bytes(recent_max_bytes)

        return await self._reme.summary_memory(
            messages=messages,
            as_llm=self.chat_model,
            as_llm_formatter=self.formatter,
            as_token_counter=get_token_counter(agent_config),
            toolkit=self.summary_toolkit,
            language=agent_config.language,
            max_input_length=agent_config.running.max_input_length,
            compact_ratio=cc.memory_compact_ratio,
            timezone=load_config().user_timezone or None,
            add_thinking_block=cc.compact_with_thinking_block,
        )

    async def memory_search(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.1,
    ) -> ToolResponse:
        """Search stored memories for relevant content."""
        self._warn_if_version_mismatch()
        if self._reme is None or not getattr(self._reme, "_started", False):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="ReMe is not started, report github issue!",
                    ),
                ],
            )
        return await self._reme.memory_search(
            query=query,
            max_results=max_results,
            min_score=min_score,
        )

    # ------------------------------------------------------------------
    # Dream-based memory optimization
    # ------------------------------------------------------------------

    async def dream_memory(self, **kwargs) -> None:
        """Run one dream-based memory optimization."""
        logger.info("running dream-based memory optimization")

        self._prepare_model_formatter()

        agent_config = load_agent_config(self.agent_id)

        set_current_workspace_dir(Path(self.working_dir))
        recent_max_bytes = (
            agent_config.running.tool_result_compact.recent_max_bytes
        )
        set_current_recent_max_bytes(recent_max_bytes)

        language = getattr(agent_config, "language", "zh")
        current_date = datetime.now().strftime("%Y-%m-%d")

        query_text = self._get_dream_prompt(
            language,
            current_date,
        )

        if not query_text.strip():
            logger.debug("dream optimization skipped: empty query")
            return

        self._prepare_model_formatter()

        self.backup_path = Path(self.working_dir).absolute() / "backup"
        self.backup_path.mkdir(parents=True, exist_ok=True)

        memory_file = Path(self.working_dir) / "MEMORY.md"
        if memory_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"memory_backup_{timestamp}.md"
            backup_file = self.backup_path / backup_filename

            try:
                shutil.copyfile(memory_file, backup_file)
                logger.info(f"Created MEMORY.md backup: {backup_file}")
            except Exception as e:
                logger.error(f"Failed to create MEMORY.md backup: {e}")
        else:
            logger.debug("No existing MEMORY.md file to backup")

        dream_agent = ReActAgent(
            name="DreamOptimizer",
            model=self.chat_model,
            sys_prompt="You are a Dream Memory Organizer specialized"
            " in optimizing long-term memory files.",
            toolkit=self.summary_toolkit,
            formatter=self.formatter,
        )

        user_msg = Msg(
            name="dream",
            role="user",
            content=[TextBlock(type="text", text=query_text)],
        )

        try:
            response = await dream_agent.reply(user_msg)
            logger.debug(
                f"Dream agent response: {response.get_text_content()}",
            )
        except Exception as e:
            logger.error("dream-based memory optimization failed: %s", repr(e))
            raise

    def _get_dream_prompt(
        self,
        language: str = "zh",
        current_date: str = "",
    ) -> str:
        """Get the dream prompt based on language."""
        return get_dream_prompt(language=language, current_date=current_date)