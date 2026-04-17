# -*- coding: utf-8 -*-
"""Memory-related prompts."""


def get_dream_prompt(language: str = "zh", current_date: str = "") -> str:
    """Get the dream prompt based on language.

    Args:
        language: Language code, 'zh' or 'en'.
        current_date: Current date string to insert into prompt.

    Returns:
        The dream memory optimization prompt.
    """
    prompts = {
        "zh": (
            "现在进入梦境状态，对长期记忆进行优化整理。请读取今日日志与现有长期记忆，"
            "在梦境中提炼高价值增量信息并去重合并，最终覆写至 `MEMORY.md`，"
            "确保长期记忆文件保持最新、精简、无冗余。\n\n"
            f"当前日期: {current_date}\n\n"
            "【梦境优化原则】\n"
            "1. 极简去冗：严禁记录流水账、Bug修复细节或单次任务。"
            "仅保留"核心业务决策"、"确认的用户偏好"与"高价值可复用经验"。"
            "2. 状态覆写：若发现状态变更（如技术栈更改、配置更新），"
            "必须用新状态替换旧状态，严禁新旧矛盾信息并存。"
            "3. 归纳整合：主动将零碎的相似规则提炼、合并为通用性强的独立条目。"
            "\n4. 废弃剔除：主动删除已被证伪的假设或不再适用的陈旧条目。\n\n"
            "【梦境执行步骤】\n步骤 1 [加载]：调用 `read` 工具，"
            "读取根目录下的 `MEMORY.md` 以及当天的日志文件 `memory/YYYY-MM-DD.md`。\n"
            "步骤 2 [梦境提纯]：在梦境中对比新旧内容，严格按照【梦境优化原则】进行去重、替换、剔除和合并，"
            "生成一份全新的记忆内容。\n步骤 3 [落盘]：调用 `write` 或 `edit` 工具，"
            "将整理后全新的 Markdown 内容覆盖写入到 `MEMORY.md` 中（请保持清晰的层级与列表结构）。"
            "\n步骤 4 [苏醒汇报]：从梦境中苏醒后，在对话中向我简短汇报：1) 新增/沉淀了哪些核心记忆；"
            "2) 修正/删除了哪些过期内容。"
        ),
        "en": (
            "Enter dream state for memory optimization. Please act as a "
            "'Dream Memory Organizer', read today's logs and existing "
            "long-term memory, extract high-value incremental information "
            "in your dream state, deduplicate and merge, and ultimately "
            "overwrite `MEMORY.md`. Ensure the long-term memory file "
            "remains up-to-date, concise, and non-redundant.\n\n"
            f"Current date: {current_date}\n\n"
            "[Dream Optimization Principles]\n1. Extreme "
            "Minimalism: Strictly forbid recording daily routines, "
            "specific bug-fix details, or one-off tasks. Retain ONLY 'core"
            " business decisions', 'confirmed user preferences', and "
            "'high-value reusable experiences'.\n2. State Overwrite: If a"
            " state change is detected (e.g., tech stack changes, config "
            "updates), you MUST replace the old state with the new one. "
            "Contradictory old and new information must not coexist.\n3. "
            "Inductive Consolidation: Proactively distill and merge "
            "fragmented, similar rules into highly universal, independent"
            " entries.\n4. Deprecation: Proactively delete hypotheses "
            "that have been proven false or outdated entries that no "
            "longer apply.\n\n[Dream Execution Steps]\nStep 1 [Load]: "
            "Invoke the `read` tool to read `MEMORY.md` in the root "
            "directory and today's log file `memory/YYYY-MM-DD.md`.\n"
            "Step 2 [Dream Purification]: Compare the old and new content "
            "in your dream state. Strictly follow the [Dream Optimization "
            "Principles] to deduplicate, replace, remove, and merge, "
            "generating entirely new memory content.\nStep 3 [Save]: "
            "Invoke the `write` or `edit` tool to overwrite the newly "
            "organized Markdown content into `MEMORY.md` (maintain clear "
            "hierarchy and list structures).\nStep 4 [Awake Report]: "
            "After waking from your dream, briefly report to me in the "
            "chat: 1) What core memories were newly added/consolidated; "
            "2) What outdated content was corrected/deleted."
        ),
    }
    return prompts.get(language, prompts["en"])


__all__ = ["get_dream_prompt"]