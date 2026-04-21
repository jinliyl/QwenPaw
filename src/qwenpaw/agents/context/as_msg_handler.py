# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches,too-many-statements
"""Handler for AgentScope message, token counting, and context."""

import json
import logging

from agentscope.message import Msg

from .as_msg_stat import AsMsgStat, AsBlockStat
from ..utils.estimate_token_counter import EstimatedTokenCounter

logger = logging.getLogger(__name__)


class AsMsgHandler:
    """Handles token counting, formatting, and context
    compaction for AgentScope messages."""

    def __init__(self, token_counter: EstimatedTokenCounter):
        self._token_counter = token_counter

    async def count_str_token(self, text: str) -> int:
        """Count tokens in a string."""
        return await self._token_counter.count(text=text)

    async def _format_tool_result_output(
        self,
        output: str | list[dict],
    ) -> tuple[str, int]:
        """Convert tool result output to string."""
        if isinstance(output, str):
            return output, await self.count_str_token(output)

        textual_parts = []
        total_token_count = 0
        for block in output:
            try:
                if not isinstance(block, dict) or "type" not in block:
                    logger.warning(
                        f"Invalid block: {block}, expected a dict "
                        f"with 'type' key, skipped.",
                    )
                    continue

                block_type = block["type"]

                if block_type == "text":
                    textual_parts.append(block.get("text", ""))
                    total_token_count += await self.count_str_token(
                        textual_parts[-1],
                    )

                elif block_type in ["image", "audio", "video"]:
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        data = source.get("data", "")
                        total_token_count += len(data) // 4 if data else 10
                    else:
                        url = source.get("url", "")
                        total_token_count += (
                            await self.count_str_token(url) if url else 10
                        )
                        textual_parts.append(f"[{block_type}] {url}")

                elif block_type == "file":
                    file_path = block.get("path", "") or block.get("url", "")
                    file_name = block.get("name", file_path)
                    textual_parts.append(f"[file] {file_name}: {file_path}")
                    total_token_count += await self.count_str_token(file_path)

                else:
                    logger.warning(
                        f"Unsupported block type '{block_type}' in "
                        f"tool result, skipped.",
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to process block {block}: {e}, skipped.",
                )

        return "\n".join(textual_parts), total_token_count

    async def stat_message(self, message: Msg) -> AsMsgStat:
        """Analyze a message and generate block statistics."""
        blocks = []
        if isinstance(message.content, str):
            blocks.append(
                AsBlockStat(
                    block_type="text",
                    text=message.content,
                    token_count=await self.count_str_token(message.content),
                ),
            )
            return AsMsgStat(
                name=message.name or message.role,
                role=message.role,
                content=blocks,
                timestamp=message.timestamp or "",
                metadata=message.metadata or {},
            )

        for block in message.content:
            block_type = block.get("type", "unknown")

            if block_type == "text":
                text = block.get("text", "")
                token_count = await self.count_str_token(text)
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text=text,
                        token_count=token_count,
                    ),
                )

            elif block_type == "thinking":
                thinking = block.get("thinking", "")
                token_count = await self.count_str_token(thinking)
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text=thinking,
                        token_count=token_count,
                    ),
                )

            elif block_type in ("image", "audio", "video"):
                source = block.get("source", {})
                url = source.get("url", "")
                if source.get("type") == "base64":
                    data = source.get("data", "")
                    token_count = len(data) // 4 if data else 10
                else:
                    token_count = (
                        await self.count_str_token(url) if url else 10
                    )
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text="",
                        token_count=token_count,
                        media_url=url,
                    ),
                )

            elif block_type == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", "")
                try:
                    input_str = json.dumps(tool_input, ensure_ascii=False)
                except (TypeError, ValueError):
                    input_str = str(tool_input)
                token_count = await self.count_str_token(tool_name + input_str)
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text="",
                        token_count=token_count,
                        tool_name=tool_name,
                        tool_input=input_str,
                    ),
                )

            elif block_type == "tool_result":
                tool_name = block.get("name", "")
                output = block.get("output", "")
                (
                    formatted_output,
                    token_count,
                ) = await self._format_tool_result_output(output)
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text="",
                        token_count=token_count,
                        tool_name=tool_name,
                        tool_output=formatted_output,
                    ),
                )

            else:
                logger.warning(
                    f"Unsupported block type {block_type}, skipped.",
                )

        return AsMsgStat(
            name=message.name or message.role,
            role=message.role,
            content=blocks,
            timestamp=message.timestamp or "",
            metadata=message.metadata or {},
        )

    async def count_msgs_token(self, messages: list[Msg]) -> int:
        """Count total token count of a list of messages."""
        total = 0
        for msg in messages:
            stat = await self.stat_message(msg)
            total += stat.total_tokens
        return total

    async def format_msgs_to_str(
        self,
        messages: list[Msg],
        context_compact_threshold: int,
        include_thinking: bool = True,
    ) -> str:
        """Format list of messages to a single formatted
        string.

        Messages are processed in reverse order (newest first)
        and older messages are skipped when token count
        exceeds context_compact_threshold.

        Args:
            messages: List of Msg objects to format.
            context_compact_threshold: Maximum token count
                before skipping older messages.
            include_thinking: Whether to include thinking blocks in output.
        """
        if not messages:
            return ""

        formatted_parts: list[str] = []
        total_token_count = 0

        for i in range(len(messages) - 1, -1, -1):
            stat = await self.stat_message(messages[i])
            formatted_content = stat.format(include_thinking=include_thinking)
            content_token_count = await self.count_str_token(formatted_content)

            is_latest = i == len(messages) - 1
            if (
                not is_latest
                and total_token_count + content_token_count
                > context_compact_threshold
            ):
                logger.info(
                    f"Skipping older messages: adding "
                    f"{content_token_count} tokens would exceed "
                    f"threshold {context_compact_threshold} "
                    f"(current: {total_token_count})",
                )
                break

            if is_latest and content_token_count > context_compact_threshold:
                logger.warning(
                    f"Latest message alone ({content_token_count} tokens) "
                    f"exceeds threshold {context_compact_threshold}, "
                    f"including it anyway.",
                )

            formatted_parts.append(formatted_content)
            total_token_count += content_token_count

        formatted_parts.reverse()
        return "\n\n".join(formatted_parts)

    @staticmethod
    def validate_tool_ids_alignment(messages: list[Msg]) -> bool:
        """Check if tool_use_ids and tool_result_ids are
        properly aligned.

        Args:
            messages: List of Msg objects to validate.

        Returns:
            True if all tool_use ids have corresponding
            tool_result ids and vice versa.
        """
        tool_use_ids: set[str] = set()
        tool_result_ids: set[str] = set()

        for msg in messages:
            for block in msg.get_content_blocks("tool_use"):
                if tool_id := block.get("id"):
                    tool_use_ids.add(tool_id)
            for block in msg.get_content_blocks("tool_result"):
                if tool_id := block.get("id"):
                    tool_result_ids.add(tool_id)

        return tool_use_ids == tool_result_ids

    async def context_check(
        self,
        messages: list[Msg],
        context_compact_threshold: int,
        context_compact_reserve: int,
    ) -> tuple[list[Msg], list[Msg], bool, int, int]:
        """Check if context exceeds threshold and split
        messages accordingly.

        Only when total tokens exceed context_compact_threshold,
        messages are split into messages_to_keep (within reserve limit)
        and messages_to_compact (older messages).

        Args:
            messages: List of Msg objects to check.
            context_compact_threshold: Maximum token count
                threshold to trigger compaction.
            context_compact_reserve: Token limit for messages to keep.

        Returns:
            A tuple of (messages_to_compact, messages_to_keep,
            tools_aligned, total_tokens, keep_tokens):
            - messages_to_compact: Older messages that
              exceed reserve limit
            - messages_to_keep: Recent messages within
              the reserve limit
            - tools_aligned: Whether tool_use and tool_result
              ids are aligned in messages_to_keep
            - total_tokens: Total token count of all messages
            - keep_tokens: Token count of messages to keep
        """
        if not messages:
            return [], [], True, 0, 0

        # Calculate total tokens and stats for all messages
        msg_stats: list[tuple[Msg, AsMsgStat]] = []
        total_tokens = 0
        for msg in messages:
            stat = await self.stat_message(msg)
            msg_stats.append((msg, stat))
            total_tokens += stat.total_tokens

        # If total tokens don't exceed threshold, no split needed
        if total_tokens < context_compact_threshold:
            return [], messages, True, total_tokens, total_tokens

        # Build bidirectional indexes for tool_use and tool_result
        # tool_id -> message index where tool_use appears
        tool_id_to_use_idx: dict[str, int] = {}
        # tool_id -> message index where tool_result appears
        tool_id_to_result_idx: dict[str, int] = {}
        # message index -> list of tool_use ids in that message
        msg_idx_to_tool_use_ids: dict[int, list[str]] = {}
        # message index -> list of tool_result ids in that message
        msg_idx_to_tool_result_ids: dict[int, list[str]] = {}

        for idx, (msg, _) in enumerate(msg_stats):
            tool_use_ids_in_msg: list[str] = []
            tool_result_ids_in_msg: list[str] = []

            for block in msg.get_content_blocks("tool_use"):
                tool_id = block.get("id", "")
                if tool_id:
                    tool_id_to_use_idx[tool_id] = idx
                    tool_use_ids_in_msg.append(tool_id)

            for block in msg.get_content_blocks("tool_result"):
                tool_id = block.get("id", "")
                if tool_id:
                    tool_id_to_result_idx[tool_id] = idx
                    tool_result_ids_in_msg.append(tool_id)

            if tool_use_ids_in_msg:
                msg_idx_to_tool_use_ids[idx] = tool_use_ids_in_msg
            if tool_result_ids_in_msg:
                msg_idx_to_tool_result_ids[idx] = tool_result_ids_in_msg

        # Iterate from the end, accumulating messages
        # to keep within reserve limit
        keep_indices: set[int] = set()
        accumulated_tokens = 0

        for i in range(len(msg_stats) - 1, -1, -1):
            # Skip messages already added as dependencies
            if i in keep_indices:
                continue

            msg, stat = msg_stats[i]

            # Check if adding this message would exceed reserve limit
            if (
                accumulated_tokens + stat.total_tokens
                > context_compact_reserve
            ):
                logger.info(
                    f"Context check: adding message {i} with "
                    f"{stat.total_tokens} tokens would exceed "
                    f"reserve {context_compact_reserve} "
                    f"(current: {accumulated_tokens})",
                )
                break

            # Find dependent message indices using pre-built indexes
            # If message has tool_use, need corresponding tool_result
            # If message has tool_result, need corresponding tool_use
            dependent_indices: set[int] = set()
            extra_tokens = 0

            # Get tool_use ids in this message -> find tool_result indices
            for tool_id in msg_idx_to_tool_use_ids.get(i, []):
                if tool_id in tool_id_to_result_idx:
                    result_idx = tool_id_to_result_idx[tool_id]
                    if result_idx not in keep_indices and result_idx != i:
                        dependent_indices.add(result_idx)
                        _, dep_stat = msg_stats[result_idx]
                        extra_tokens += dep_stat.total_tokens

            # Get tool_result ids in this message -> find tool_use indices
            for tool_id in msg_idx_to_tool_result_ids.get(i, []):
                if tool_id in tool_id_to_use_idx:
                    use_idx = tool_id_to_use_idx[tool_id]
                    if use_idx not in keep_indices and use_idx != i:
                        dependent_indices.add(use_idx)
                        _, dep_stat = msg_stats[use_idx]
                        extra_tokens += dep_stat.total_tokens

            # Check if we can fit this message plus dependencies
            if (
                accumulated_tokens + stat.total_tokens + extra_tokens
                > context_compact_reserve
            ):
                has_tool_use = i in msg_idx_to_tool_use_ids
                has_tool_result = i in msg_idx_to_tool_result_ids
                dep_type = (
                    "tool_result"
                    if has_tool_use
                    else "tool_use"
                    if has_tool_result
                    else "unknown"
                )
                logger.info(
                    f"Context check: message {i} requires "
                    f"{extra_tokens} extra tokens for {dep_type} "
                    f"dependencies, total would exceed reserve "
                    f"{context_compact_reserve}",
                )
                break

            # Add this message and its dependencies
            keep_indices.add(i)
            keep_indices.update(dependent_indices)
            accumulated_tokens += stat.total_tokens + extra_tokens

        # Build final lists based on keep_indices (preserve original order)
        messages_to_compact = []
        messages_to_keep = []

        for idx, (msg, _) in enumerate(msg_stats):
            if idx in keep_indices:
                messages_to_keep.append(msg)
            else:
                messages_to_compact.append(msg)

        # Validate tool ids alignment for messages_to_keep
        tools_aligned = self.validate_tool_ids_alignment(messages_to_keep)

        keep_tokens = accumulated_tokens

        logger.info(
            f"Context check result: {len(messages_to_compact)} "
            f"messages to compact, {len(messages_to_keep)} "
            f"messages to keep, total tokens: {total_tokens}, "
            f"threshold: {context_compact_threshold}, "
            f"reserve: {context_compact_reserve}, "
            f"kept tokens: {keep_tokens}, "
            f"tools_aligned: {tools_aligned}",
        )

        return (
            messages_to_compact,
            messages_to_keep,
            tools_aligned,
            total_tokens,
            keep_tokens,
        )
