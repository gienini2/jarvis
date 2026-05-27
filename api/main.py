import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from core.llm import generate
from core.memory import save_message
from core.intent import classify, log_intent
from core.tasks import (
    resolve_task_ref,
    complete_task,
    find_tasks_by_project,
    find_tasks_by_keyword,
    sync_tasks_from_json,
    add_task_context,
)
from core.dashboard import get_pending_tasks

os.environ["PYTHONUTF8"] = "1"

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


class ContextRequest(BaseModel):
    mando_key: str
    type: str       # note | decision | file | link
    content: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat — entry point principal
# ---------------------------------------------------------------------------

@app.post("/chat")
def chat(req: ChatRequest):
    save_message("user", req.message)

    intent = classify(req.message)

    if intent.type == "complete_task" and intent.task_ref:
        response = _handle_complete_task(intent.task_ref)
        log_intent(req.message, "complete_task", {"task_ref": intent.task_ref}, executed=True)

    elif intent.type == "query_task":
        response = _handle_query_task(intent.project_ref or "", req.message)
        log_intent(req.message, "query_task", {"project_ref": intent.project_ref})

    else:
        response = generate(req.message)
        log_intent(req.message, intent.type, intent.payload or None)

    save_message("assistant", response)

    return JSONResponse(
        content={"response": response, "intent": intent.type},
        media_type="application/json; charset=utf-8",
    )


def _handle_complete_task(task_ref: str) -> str:
    matches = resolve_task_ref(task_ref)

    if not matches:
        return f"No encontré ninguna tarea pendiente con '{task_ref}'."

    if len(matches) == 1:
        t = matches[0]
        ok = complete_task(t["mando_key"])
        if ok:
            return f"Tarea marcada como completada en el Centro de Mando:\n[{t['priority']}] {t['project']} → {t['title']}"
        return "No pude actualizar el estado de la tarea."

    # Multiple matches — list them for clarification
    lines = "\n".join(
        f"  {i+1}. [{t['priority']}] {t['mando_key']} — {t['title']}"
        for i, t in enumerate(matches[:5])
    )
    return f"Encontré varias tareas con '{task_ref}'. ¿Cuál quieres marcar como hecha?\n{lines}"


def _handle_query_task(project_ref: str, original_message: str) -> str:
    # Try by project_id first, fall back to keyword search
    tasks = find_tasks_by_project(project_ref) if project_ref else []
    if not tasks:
        tasks = find_tasks_by_keyword(project_ref)

    if not tasks:
        # No specific project found — answer with LLM + active memory
        return generate(original_message)

    lines = "\n".join(
        f"  [{t['priority']}] {t['title']}"
        for t in tasks[:10]
    )
    header = f"Tareas pendientes — {tasks[0]['project']}:\n"
    return header + lines


# ---------------------------------------------------------------------------
# Tasks API
# ---------------------------------------------------------------------------

@app.get("/tasks")
def get_tasks(
    project: str = Query(None, description="project_id"),
    keyword: str = Query(None),
    status: str = Query(None),
):
    if project:
        tasks = find_tasks_by_project(project, exclude_done=(status != "all"))
    elif keyword:
        tasks = find_tasks_by_keyword(keyword, exclude_done=(status != "all"))
    else:
        tasks = get_pending_tasks(limit=30)
    return tasks


@app.post("/tasks/sync")
def tasks_sync():
    n = sync_tasks_from_json()
    return {"synced": n}


@app.post("/tasks/context")
def tasks_add_context(req: ContextRequest):
    ok = add_task_context(req.mando_key, req.type, req.content)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "task not found"})
    return {"status": "ok"}
