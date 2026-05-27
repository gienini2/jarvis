from core.llm import generate, generate_stream
from core.memory import save_message
import sys

sys.stdout.reconfigure(encoding="utf-8")

print("Jarvis listo (RTX 5050 GPU)")
print("Escribe 'exit' para salir\n")


while True:

    print("Juanpa: (poner END al final de todo)")

    lines = []

    while True:

        line = input()

        if line.strip() == "END":
            break

        lines.append(line)

    user_input = "\n".join(lines)

    if user_input.lower() == "exit":
        break

    save_message("user", user_input)

    print("\nJarvis: ", end="", flush=True)
    response = generate_stream(user_input)
    print("\n")

    save_message("assistant", response)