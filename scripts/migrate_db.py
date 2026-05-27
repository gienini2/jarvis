"""
Migración: crea tablas nuevas y carga tareas desde dashboard-data.json.
Idempotente — se puede volver a ejecutar sin romper nada.
Usa SERIAL (integer) para PKs: sin dependencias de extensiones UUID.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import get_connection


def run():
    conn = get_connection()
    cur = conn.cursor()

    # Verificar si tasks tiene el esquema correcto (columna mando_key)
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'tasks' AND column_name = 'mando_key'
    """)
    has_mando_key = cur.fetchone() is not None
    if not has_mando_key:
        print("Tabla tasks existe con esquema antiguo — recreando...")
        cur.execute("DROP TABLE IF EXISTS task_context CASCADE")
        cur.execute("DROP TABLE IF EXISTS tasks CASCADE")

    print("Creando tabla tasks...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id          SERIAL PRIMARY KEY,
            mando_key   TEXT UNIQUE,
            title       TEXT NOT NULL,
            status      TEXT DEFAULT 'pending',
            priority    TEXT DEFAULT 'mid',
            project     TEXT,
            project_id  TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            updated_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    print("Creando tabla task_context...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS task_context (
            id          SERIAL PRIMARY KEY,
            task_id     INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
            type        TEXT NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    print("Creando tabla intents...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS intents (
            id          SERIAL PRIMARY KEY,
            message     TEXT NOT NULL,
            intent_type TEXT,
            payload     JSONB,
            executed    BOOLEAN DEFAULT FALSE,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    print("Añadiendo columnas a conversations...")
    cur.execute("""
        ALTER TABLE conversations
        ADD COLUMN IF NOT EXISTS task_id    INTEGER,
        ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Tablas listas.")

    print("Sincronizando tareas desde dashboard-data.json...")
    from core.tasks import sync_tasks_from_json
    n = sync_tasks_from_json()
    print(f"  {n} tareas sincronizadas.")
    print("Migración completada.")


if __name__ == "__main__":
    run()
