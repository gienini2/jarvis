import re
import json
from dataclasses import dataclass, field
from typing import Optional

from core.db import get_connection

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Intent:
    type: str                           # complete_task | create_task | query_task | general
    task_ref: Optional[str] = None
    project_ref: Optional[str] = None
    payload: dict = field(default_factory=dict)
    confidence: float = 1.0


# Maps shorthand mention → project_id in dashboard-data.json
PROJECT_ALIASES = {
    "dragai": "dragai-tech",
    "drag": "dragai-tech",
    "bitacola": "dragai-tech",
    "bitàcola": "dragai-tech",
    "figaro": "figaro-policial",
    "figaró": "figaro-policial",
    "arboc": "arboc-policial",
    "arboç": "arboc-policial",
    "betatesters": "betatesters",
    "acoso": "acoso-laboral",
    "laboral": "acoso-laboral",
    "juicio": "juicio-pension",
    "pension": "juicio-pension",
    "pensión": "juicio-pension",
    "renta": "declaracion-renta",
    "aeat": "declaracion-renta",
    "mossos": "opos-mossos",
    "castellbisbal": "opos-castellbisbal",
    "mataro": "opos-mataro",
    "mataró": "opos-mataro",
    "prat": "opos-prat",
    "calafell": "opos-calafell",
    "terrassa": "opos-terrassa",
    "delta": "delta",
    "cursos": "cursos-chatgpt",
    "sinergia": "cursos-chatgpt",
    "tingbot": "tingbot",
    "noctorial": "fondeo-noctorial",
    "cft": "fondeo-cft",
    "wsf": "fondeo-wsf",
    "trading": "trading-general",
    "fondeo": None,   # ambiguous — matches multiple
    "opos": None,     # ambiguous
}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_COMPLETE = [
    r"(?:marca[r]?|pon[er]*|marcar)\s+(?:como\s+)?(?:hech[oa]|complet[ao]d[ao]|done|listo|finalizado)\s+(.+)",
    r"(.+?)\s+(?:está|esta|queda|ya\s+(?:está|esta))\s+(?:hech[oa]|complet[ao]d[ao]|done|listo|finalizado|terminad[ao])",
    r"(?:complet[ao]r?|finaliz[ao]r?|cerrar?|done)\s+(.+)",
]

_CREATE = [
    r"(?:recuérdame|recuérdate|recuerda(?:me)?)\s+(.+)",
    r"(?:añade?|agrega?|crea?)\s+(?:una?\s+)?(?:tarea|nota|recordatorio)[:\s]+(.+)",
    r"(?:nueva\s+tarea|nuevo\s+recordatorio)[:\s]+(.+)",
    r"(?:apunta|apúntate)\s+(?:que\s+)?(.+)",
]

_QUERY = [
    r"(?:qué|que)\s+(?:hay|tengo|está|esta|queda)\s+(?:pendiente|en\s+curso|activo)\s+(?:en|para|de|sobre)?\s*(.+)",
    r"(?:estado|status)\s+(?:de[l]?|en)?\s+(.+)",
    r"(?:tareas|pendientes|agenda)\s+(?:de[l]?|en|para)?\s*(.+)",
    r"(?:muéstrame|muestra(?:me)?|ver|dame)\s+(?:las?\s+)?(?:tareas|pendientes)\s+(?:de[l]?|en)?\s*(.+)",
    r"(?:cómo|como)\s+(?:va|está|esta|vamos)\s+(?:con\s+)?(?:lo\s+de\s+)?(.+)",
]

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(message: str) -> Intent:
    msg = message.strip()

    for pattern in _COMPLETE:
        m = re.search(pattern, msg, re.IGNORECASE)
        if m:
            return Intent(type="complete_task", task_ref=m.group(1).strip(), confidence=0.9)

    for pattern in _CREATE:
        m = re.search(pattern, msg, re.IGNORECASE)
        if m:
            return Intent(
                type="create_task",
                payload={"text": m.group(1).strip()},
                confidence=0.9,
            )

    for pattern in _QUERY:
        m = re.search(pattern, msg, re.IGNORECASE)
        if m:
            raw = m.group(1).strip().rstrip("?.,!")
            project_id = _resolve_project(raw)
            return Intent(
                type="query_task",
                project_ref=project_id or raw,
                confidence=0.85,
            )

    return Intent(type="general", confidence=1.0)


def _resolve_project(raw: str) -> Optional[str]:
    key = raw.lower().strip()
    return PROJECT_ALIASES.get(key)


# ---------------------------------------------------------------------------
# Intent log
# ---------------------------------------------------------------------------

def log_intent(message: str, intent_type: str, payload: dict = None, executed: bool = False):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO intents (message, intent_type, payload, executed)
        VALUES (%s, %s, %s, %s)
        """,
        (message, intent_type, json.dumps(payload) if payload else None, executed),
    )
    conn.commit()
    cur.close()
    conn.close()
