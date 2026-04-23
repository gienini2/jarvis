"""
memory/db.py
============
Gestión de conexión a PostgreSQL + pgvector.
Usa un pool de conexiones para robustez en producción.
"""

import os
import logging
from psycopg2 import pool
from pgvector.psycopg2 import register_vector

logger = logging.getLogger("jarvis.db")

# ── Configuración desde variables de entorno (recomendado) o fallback ─────────
DB_CONFIG = {
    "dbname":   os.getenv("JARVIS_DB_NAME",     "jarvis"),
    "user":     os.getenv("JARVIS_DB_USER",     "jarvis_user"),
    "password": os.getenv("JARVIS_DB_PASSWORD", "cambia_esta_password"),
    "host":     os.getenv("JARVIS_DB_HOST",     "127.0.0.1"),
    "port":     os.getenv("JARVIS_DB_PORT",     "5432"),
}

_pool: pool.SimpleConnectionPool | None = None


def _get_pool() -> pool.SimpleConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        logger.info("Creando pool de conexiones PostgreSQL...")
        _pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            **DB_CONFIG,
        )
        # Registrar pgvector en todas las conexiones del pool
        conn = _pool.getconn()
        register_vector(conn)
        _pool.putconn(conn)
        logger.info("Pool creado correctamente.")
    return _pool


def get_connection():
    """
    Devuelve una conexión del pool.
    IMPORTANTE: llama a release_connection(conn) cuando termines.
    """
    return _get_pool().getconn()


def release_connection(conn):
    """Devuelve la conexión al pool."""
    try:
        _get_pool().putconn(conn)
    except Exception as e:
        logger.warning("Error devolviendo conexión al pool: %s", e)


def close_pool():
    """Cierra todas las conexiones. Llamar al apagar la app."""
    global _pool
    if _pool and not _pool.closed:
        _pool.closeall()
        logger.info("Pool de conexiones cerrado.")
