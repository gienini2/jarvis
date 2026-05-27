import json
import requests

from config.settings import settings
from core.active_memory import get_context

SYSTEM_PROMPT = """Eres Jarvis.

Asistente técnico personal de Juan Pablo.

Prioridades:
- precisión técnica
- continuidad contextual
- respuestas claras y directas
- memoria operativa
- pensamiento estructurado

La fuente de verdad de tareas es el Centro de Mando.

Si el usuario pregunta por tareas pendientes:
- responder únicamente tareas no completadas
- no inventar tareas
- no mostrar tareas finalizadas
"""


def build_prompt(user_prompt: str) -> str:
    context = get_context(user_prompt)
    return (
        SYSTEM_PROMPT
        + "\n\n"
        + context
        + f"\n\nuser: {user_prompt}\nassistant:"
    )


def generate(prompt: str) -> str:
    final_prompt = build_prompt(prompt)

    response = requests.post(
        settings.OLLAMA_URL,
        json={
            "model": settings.MODEL,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "num_gpu": 99,
                "num_ctx": 4096,
                "temperature": 0.2,
            },
        },
        timeout=settings.TIMEOUT,
    )

    return response.json()["response"]


def generate_stream(prompt: str) -> str:
    """Igual que generate() pero imprime tokens en tiempo real. Devuelve el texto completo."""
    final_prompt = build_prompt(prompt)

    response = requests.post(
        settings.OLLAMA_URL,
        json={
            "model": settings.MODEL,
            "prompt": final_prompt,
            "stream": True,
            "options": {
                "num_gpu": 99,
                "num_ctx": 4096,
                "temperature": 0.2,
            },
        },
        timeout=settings.TIMEOUT,
        stream=True,
    )

    full_response = []
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("response", "")
            print(token, end="", flush=True)
            full_response.append(token)
            if chunk.get("done"):
                break

    return "".join(full_response)
