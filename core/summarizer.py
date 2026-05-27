from core.memory import get_recent_messages
from core.llm import generate
from core.db import get_connection


def create_summary():

    messages = get_recent_messages(limit=20)

    text = ""

    for role, message in messages:
        text += f"{role}: {message}\n"

    prompt = f"""
Resume esta conversación de forma técnica y compacta.

Objetivos:
- conservar contexto importante
- conservar decisiones
- conservar datos personales relevantes
- eliminar ruido

Conversación:

{text}

Resumen:
"""

    summary = generate(prompt)

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO summaries (content)
        VALUES (%s)
        """,
        (summary,)
    )

    conn.commit()

    cur.close()
    conn.close()

    return summary