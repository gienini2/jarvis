from core.db import get_connection


def save_message(role: str, message: str, task_id=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO conversations (role, message, task_id)
        VALUES (%s, %s, %s)
        """,
        (role, message, task_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_recent_messages(limit=10):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, message
        FROM conversations
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    rows.reverse()
    return rows


def get_recent_summaries(limit=3):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content
        FROM summaries
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    rows.reverse()
    return [row[0] for row in rows]
