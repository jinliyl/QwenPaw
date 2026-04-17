import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentscope.agent import ReActAgent
from agentscope.message import Msg, TextBlock
from qwenpaw.agents.tools.utils import truncate_text_output, DEFAULT_MAX_BYTES, TRUNCATION_NOTICE_MARKER
from qwenpaw.constant import MEMORY_COMPACT_KEEP_RECENT

from .as_msg_handler import AsMsgHandler
from .base_context_management import cm_registry, BaseContextManagement
from ...config import load_agent_config
from ...agents.utils import check_valid_messages, get_token_counter
from .estimate_token_counter import EstimatedTokenCounter

if TYPE_CHECKING:
    from ..memory import BaseLongTermMemoryService

logger = logging.getLogger(__name__)


@cm_registry.register("light")
class LightContextManagement(BaseContextManagement):
    """Lightweight context management for message compaction.

    This class manages context size by checking if messages exceed the
    token threshold and compacting older messages when needed.

    Attributes:
        memory_compact_threshold (int): Token count threshold for triggering compaction.
        memory_compact_reserve (int): Token count to reserve for recent messages.
        messages (list[Msg]): List of conversation messages to manage.
        token_counter: Token counter instance for counting tokens.
        tool_result_dir (Path): Directory to save truncated tool result content.
        retention_days (int): Number of days to retain tool result files.
        old_max_bytes (int): Byte limit for old tool results.
        recent_max_bytes (int): Byte limit for recent tool results.
        recent_n (int): Number of recent messages to use recent_max_bytes for.
        encoding (str): Character encoding for truncation.
        compact_with_thinking_block (bool): Whether to include thinking blocks in compaction.
        previous_summary (str): Previous compaction summary for incremental updates.
        memory_manager (BaseLongTermMemoryService): Memory manager instance for compaction.
    """

    _REENTRANCY_ATTR = "_memory_compact_hook_running"

    def __init__(self, agent_id: str, working_dir: str | Path, memory_manager: "BaseLongTermMemoryService | None" = None):
        super().__init__(agent_id=agent_id, working_dir=working_dir)
        self.memory_manager: "BaseLongTermMemoryService | None" = memory_manager
        self.tool_result_dir: Path = self.working_dir / "tool_results"
        self.retention_days: int = 5
        self.old_max_bytes: int = 3000
        self.recent_max_bytes: int = DEFAULT_MAX_BYTES
        self.recent_n: int = 2
        self.encoding: str = "utf-8"
        self.compact_with_thinking_block: bool = True
        self.previous_summary: str = ""
        self.token_counter: EstimatedTokenCounter | None = None
        self.memory_compact_threshold: int = 0
        self.memory_compact_reserve: int = 0
        self.messages: list[Msg] = []
        self.tool_result_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    async def _print_status_message(
        agent: ReActAgent,
        text: str,
    ) -> None:
        """Print a status message to the agent's output.

        Args:
            agent: The agent instance to print the message for.
            text: The text content of the status message.
        """
        msg = Msg(
            name=agent.name,
            role="assistant",
            content=[TextBlock(type="text", text=text)],
        )
        await agent.print(msg)

    def _truncate(self, content: str, max_bytes: int) -> str:
        if not content:
            return content

        try:
            if TRUNCATION_NOTICE_MARKER in content:
                return truncate_text_output(content, max_bytes=max_bytes, encoding=self.encoding)

            if len(content.encode(self.encoding)) <= max_bytes + 100:
                return content

            saved_path: str | None = None
            fp = self.tool_result_dir / f"{uuid.uuid4().hex}.txt"
            fp.write_text(content, encoding=self.encoding)
            saved_path = str(fp)

            return truncate_text_output(
                content,
                1,
                content.count("\n") + 1,
                max_bytes,
                file_path=saved_path,
                encoding=self.encoding,
            )
        except Exception as e:
            logger.warning("Failed to truncate content, returning original: %s", e)
            return content

    def _compact_output(self, output: str | list[dict], max_bytes: int) -> str | list[dict]:
        """Truncate output to max_bytes, saving full content to file if needed."""

        if isinstance(output, str):
            return self._truncate(output, max_bytes)
        if isinstance(output, list):
            for b in output:
                if isinstance(b, dict) and b.get("type") == "text":
                    b["text"] = self._truncate(b.get("text", ""), max_bytes)
        return output

    def cleanup_expired_files(self) -> int:
        """Clean up files older than retention_days.

        Returns:
            Number of files successfully deleted.
        """
        if not self.tool_result_dir.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=self.retention_days)
        deleted = failed = 0

        for fp in self.tool_result_dir.glob("*.txt"):
            try:
                stat = os.stat(fp)
                if sys.platform == "win32":
                    ts = stat.st_ctime  # creation time on Windows
                else:
                    ts = getattr(stat, "st_birthtime", stat.st_mtime)  # macOS/BSD; Linux fallback to mtime
                if datetime.fromtimestamp(ts) < cutoff:
                    fp.unlink()
                    deleted += 1
            except FileNotFoundError:
                pass  # deleted by another process between glob and stat/unlink
            except Exception as e:
                failed += 1
                logger.warning("Failed to delete %s: %s", fp, e)

        if deleted or failed:
            logger.info("Cleaned up %d expired files (%d failed)", deleted, failed)
        return deleted

    async def _check_context(self, messages: list[Msg]) -> tuple[list[Msg], list[Msg], bool]:
        """Check context size and determine if compaction is needed.

        Checks if messages exceed the token threshold. If so, splits them
        into messages to compact and messages to keep.

        Args:
            messages: List of conversation messages to check.

        Returns:
            tuple[list[Msg], list[Msg], bool]: A tuple containing:
                - messages_to_compact: Older messages that should be compacted.
                - messages_to_keep: Recent messages to keep in context.
                - is_valid: True if tool calls aligned, False otherwise.

        Note:
            - Returns ([], messages, True) if no compaction is needed.
            - is_valid=False indicates tool_use and tool_result are misaligned.
        """
        if not messages:
            return [], [], True

        if self.token_counter is None:
            return [], messages, True

        msg_handler = AsMsgHandler(self.token_counter)

        messages_to_compact, messages_to_keep, is_valid = await msg_handler.context_check(
            messages=messages,
            context_compact_threshold=self.memory_compact_threshold,
            context_compact_reserve=self.memory_compact_reserve,
        )

        if messages_to_compact:
            logger.info(
                f"LightContextManagement Result: "
                f"to_compact={len(messages_to_compact)}, "
                f"to_keep={len(messages_to_keep)}, "
                f"is_valid={is_valid}",
            )

        return messages_to_compact, messages_to_keep, is_valid

    async def _compact(self, messages_to_compact: list[Msg]) -> str:
        """Compact messages into a structured summary.

        Args:
            messages_to_compact: Older messages that should be compacted.

        Returns:
            Structured summary string, or empty string if no messages to compact.
        """
        if not messages_to_compact:
            return self.previous_summary

        if self.token_counter is None:
            return self.previous_summary

        msg_handler = AsMsgHandler(self.token_counter)
        history_formatted_str: str = await msg_handler.format_msgs_to_str(
            messages=messages_to_compact,
            context_compact_threshold=self.memory_compact_threshold,
            include_thinking=self.compact_with_thinking_block,
        )

        if not history_formatted_str:
            logger.warning(f"No history to compact. messages={len(messages_to_compact)}")
            return self.previous_summary

        # For now, return the formatted string as the summary
        # In a full implementation, this would call an LLM to generate a structured summary
        logger.info(f"Compacted {len(messages_to_compact)} messages into summary")

        # Update previous summary
        if self.previous_summary:
            self.previous_summary = f"{self.previous_summary}\n\n--- New Context ---\n\n{history_formatted_str}"
        else:
            self.previous_summary = history_formatted_str

        return self.previous_summary

    async def pre_reply(
        self,
        agent: ReActAgent,
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Pre-reply hook. Currently no implementation."""
        return None

    async def post_reply(
        self,
        agent: ReActAgent,
        kwargs: dict[str, Any],
        output: Msg,
    ) -> Msg | None:
        """Post-reply hook. Currently no implementation."""
        return None

    async def pre_reasoning(
        self,
        agent: ReActAgent,
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Process messages before reasoning phase.

        This hook extracts system prompt messages and recent messages,
        builds an estimated full context prompt, and triggers compaction
        when the total estimated token count exceeds the threshold.

        Memory structure:
            [System Prompt (preserved)] + [Compactable (counted)] +
            [Recent (preserved)]

        Args:
            agent: The agent instance.
            kwargs: Input arguments to the _reasoning method.

        Returns:
            None (hook doesn't modify kwargs)
        """
        messages: list[Msg] = kwargs.get("messages", [])

        # Guard against duplicate execution caused by the metaclass
        # wrapping _reasoning at multiple levels of the class hierarchy
        if getattr(agent, self._REENTRANCY_ATTR, False):
            return None
        setattr(agent, self._REENTRANCY_ATTR, True)

        try:
            # Get hot-reloaded agent config
            agent_config = load_agent_config(self.agent_id)
            running_config = agent_config.running
            token_counter = get_token_counter(agent_config)

            # Update context management settings from config
            self.token_counter = token_counter
            self.memory_compact_threshold = running_config.memory_compact_threshold
            self.memory_compact_reserve = running_config.memory_compact_reserve

            memory = agent.memory
            memory_manager = self.memory_manager

            system_prompt = agent.sys_prompt
            compressed_summary = memory.get_compressed_summary() if memory else ""
            str_token_count = await token_counter.count(
                messages=[],
                text=(system_prompt or "") + (compressed_summary or ""),
            )

            # Update threshold to account for system prompt and compressed summary
            self.memory_compact_threshold = (
                running_config.memory_compact_threshold - str_token_count
            )

            if self.memory_compact_threshold <= 0:
                logger.warning(
                    "The memory_compact_threshold is set too low; "
                    "the combined token length of system_prompt and "
                    "compressed_summary exceeds the configured threshold. "
                    "Alternatively, you could use /clear to reset the context "
                    "and compressed_summary, ensuring the total remains "
                    "below the threshold.",
                )
                return None

            if memory:
                messages = await memory.get_memory(prepend_summary=False)

            # memory_compact_reserve is always available from config
            (
                messages_to_compact,
                messages_to_keep,
                is_valid,
            ) = await self._check_context(messages=messages)

            if not messages_to_compact:
                return None

            if not is_valid:
                logger.warning(
                    "Please include the output of the /history command when "
                    "reporting the bug to the community. Invalid "
                    "messages=%s",
                    messages,
                )
                keep_length: int = MEMORY_COMPACT_KEEP_RECENT
                messages_length = len(messages)
                while keep_length > 0 and not check_valid_messages(
                    messages[max(messages_length - keep_length, 0) :],
                ):
                    keep_length -= 1

                if keep_length > 0:
                    messages_to_compact = messages[
                        : max(messages_length - keep_length, 0)
                    ]
                else:
                    messages_to_compact = messages

            if not messages_to_compact:
                return None

            if running_config.memory_summary.memory_summary_enabled and memory_manager:
                memory_manager.add_async_summary_task(
                    messages=messages_to_compact,
                )

            await self._print_status_message(
                agent,
                "🔄 Context compaction started...",
            )

            compact_content = ""
            if running_config.context_compact.context_compact_enabled:
                compact_content = await self._compact(messages_to_compact=messages_to_compact)
                if not compact_content:
                    await self._print_status_message(
                        agent,
                        "⚠️ Context compaction failed.",
                    )
                else:
                    await self._print_status_message(
                        agent,
                        "✅ Context compaction completed",
                    )
            else:
                await self._print_status_message(
                    agent,
                    "✅ Context compaction skipped",
                )

            if memory:
                updated_count = await memory.mark_messages_compressed(
                    messages_to_compact,
                )
                logger.info(f"Marked {updated_count} messages as compacted")

                await memory.update_compressed_summary(compact_content)

            self.messages = messages_to_keep
            return None

        except Exception as e:
            logger.exception(
                "Failed to compact memory in pre_reasoning hook: %s",
                e,
                exc_info=True,
            )
            return None

        finally:
            setattr(agent, self._REENTRANCY_ATTR, False)

    async def _pruning_tool_result(self, messages: list[Msg]) -> list[Msg]:
        """Process all messages, truncating large tool results.

        Args:
            messages: List of conversation messages to process.

        Returns:
            Processed messages with truncated tool results.
        """
        if not messages:
            return messages

        recent_n = 0
        for msg in reversed(messages):
            if not isinstance(msg.content, list) or not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in msg.content
            ):
                break
            recent_n += 1
        split_index = max(0, len(messages) - max(recent_n, self.recent_n))

        md_file_tool_ids = set()
        try:
            for msg in messages:
                if not isinstance(msg.content, list):
                    continue

                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_id = block.get("id", "")
                        if not tool_id:
                            continue

                        if (
                            block.get("name", "").lower() == "read_file"
                            and ".md" in (block.get("raw_input") or "").lower()
                        ):
                            md_file_tool_ids.add(tool_id)
        except Exception as e:
            logger.warning("Failed to detect md file tool ids: %s", e)
        logger.info(f"md_file_tool_ids: {md_file_tool_ids}")

        for idx, msg in enumerate(messages):
            if not isinstance(msg.content, list):
                continue
            is_recent = idx >= split_index
            max_bytes = self.recent_max_bytes if is_recent else self.old_max_bytes
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("output"):
                    tool_use_id = block.get("id", "")
                    if tool_use_id in md_file_tool_ids:
                        effective_max_bytes = self.recent_max_bytes
                    else:
                        effective_max_bytes = max_bytes
                    block["output"] = self._compact_output(block["output"], effective_max_bytes)

        self.cleanup_expired_files()
        return messages

    async def post_acting(
        self,
        agent: ReActAgent,
        kwargs: dict[str, Any],
        output: Any,
    ) -> Msg | None:
        """Process messages after action, compacting tool results if enabled.

        Args:
            agent: The agent instance.
            kwargs: Input arguments to the _acting method.
            output: The output from the _acting method.

        Returns:
            None (hook doesn't modify output)
        """
        messages: list[Msg] = kwargs.get("messages", [])
        if not messages:
            return None

        agent_config = load_agent_config(self.agent_id)
        running_config = agent_config.running

        # Compact tool results with configured thresholds
        trc = running_config.tool_result_compact
        if trc.enabled:
            self.recent_n = trc.recent_n
            self.old_max_bytes = trc.old_max_bytes
            self.recent_max_bytes = trc.recent_max_bytes
            self.retention_days = trc.retention_days
            await self._pruning_tool_result(messages=messages)
            return None

        # Fallback to local pruning config
        cm_config = agent_config.running.light_context_management_config
        tool_result_pruning_config = cm_config.tool_result_pruning_config

        if tool_result_pruning_config.enabled:
            self.recent_n = tool_result_pruning_config.pruning_recent_n
            self.old_max_bytes = tool_result_pruning_config.pruning_old_msg_max_bytes
            self.recent_max_bytes = tool_result_pruning_config.pruning_recent_msg_max_bytes
            self.retention_days = tool_result_pruning_config.offload_retention_days

            await self._pruning_tool_result(messages=messages)

        return None
