import json
import requests
import urllib3
from pathlib import Path

urllib3.disable_warnings()

ESTADO_URL = "https://figaro-server.taild7b9e6.ts.net/mando-api/estado"
DATA_FILE = Path(__file__).parent.parent / "dashboard-data.json"


def load_data() -> list:
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_estado() -> dict:
    try:
        res = requests.get(ESTADO_URL, timeout=5, verify=False)
        res.raise_for_status()
        return res.json()
    except Exception:
        return {}


def post_estado(estado: dict) -> bool:
    try:
        res = requests.post(
            ESTADO_URL,
            json=estado,
            timeout=5,
            verify=False,
            headers={"Content-Type": "application/json"},
        )
        return res.ok
    except Exception:
        return False


def get_dashboard() -> list:
    data = load_data()
    estado = get_estado()
    for project in data:
        for i, task in enumerate(project["tasks"]):
            key = f'{project["id"]}_{i}'
            task["done"] = estado.get(key, False)
    return data


def get_pending_tasks(limit: int = 15) -> list:
    dashboard = get_dashboard()
    pendientes = []
    for project in dashboard:
        for task in project["tasks"]:
            if not task.get("done"):
                pendientes.append({
                    "project": project["title"],
                    "project_id": project["id"],
                    "text": task["text"],
                    "priority": task["priority"],
                })
    prioridad = {"high": 0, "mid": 1, "low": 2}
    pendientes.sort(key=lambda x: prioridad.get(x["priority"], 99))
    return pendientes[:limit]
