import re

from core.memory import get_recent_messages, get_recent_summaries
from core.tasks import find_tasks_by_keyword, find_tasks_by_project, get_task_context
from core.dashboard import get_pending_tasks
from core.intent import PROJECT_ALIASES

# Keywords that map directly to a project_id
_KNOWN_KEYWORDS = set(PROJECT_ALIASES.keys())


def get_context(user_message: str) -> str:
    """
    Build an active context string tailored to the user message.
    Only injects relevant information — not the full memory dump.
    """
    parts = []

    # 1. Consolidated memory (one recent summary)
    summaries = get_recent_summaries(limit=1)
    if summaries:
        parts.append("MEMORIA CONSOLIDADA:\n" + summaries[0])

    # 2. Task-specific context derived from the message
    keywords = _extract_keywords(user_message)
    if keywords:
        seen_keys = set()
        task_lines = []

        for kw in keywords[:4]:
            # Try direct project lookup first
            project_id = PROJECT_ALIASES.get(kw.lower())
            if project_id:
                tasks = find_tasks_by_project(project_id)
            else:
                tasks = find_tasks_by_keyword(kw)

            for t in tasks[:5]:
                if t["mando_key"] not in seen_keys:
                    seen_keys.add(t["mando_key"])
                    task_lines.append(
                        f"  [{t['priority']}] {t['project']} → {t['title']}"
                    )

                    # Attach any stored context notes for this task
                    ctx = get_task_context(t["mando_key"])
                    for c in ctx[:2]:
                        task_lines.append(f"    [{c['type']}] {c['content']}")

        if task_lines:
            parts.append("CONTEXTO DE TAREAS RELEVANTES:\n" + "\n".join(task_lines))

    # 3. High-priority pending tasks (always, limited)
    pending = get_pending_tasks(limit=8)
    if pending:
        lines = "\n".join(
            f"  [{t['priority']}] [{t['project']}] {t['text']}"
            for t in pending
        )
        parts.append("TAREAS PRIORITARIAS:\n" + lines)

    # 4. Recent conversation (last 5 turns)
    messages = get_recent_messages(limit=5)
    if messages:
        conv = "\n".join(f"{role}: {msg}" for role, msg in messages)
        parts.append("CONVERSACIÓN RECIENTE:\n" + conv)

    return "\n\n".join(parts)


def _extract_keywords(message: str) -> list:
    """Extract task codes and project keywords from a message."""
    # Task codes: A-13, AA-31, D-10, AP-01, D-OM-1
    codes = re.findall(r'\b[A-Z]{1,2}-(?:OM-)?\d+[ab]?\b', message, re.IGNORECASE)
    # Known project keywords
    words = [w.lower().rstrip(".,!?") for w in message.split()]
    matched = [w for w in words if w in _KNOWN_KEYWORDS]
    return list(dict.fromkeys(codes + matched))  # preserve order, deduplicate
