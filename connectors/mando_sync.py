import json
import requests
import urllib3
import re
import json5

from core.db import get_connection

urllib3.disable_warnings()

BASE_URL = "https://figaro-server.taild7b9e6.ts.net"

ESTADO_URL = f"{BASE_URL}/mando-api/estado"


def load_remote_dashboard():

    response = requests.get(
        f"{BASE_URL}/mando/",
        verify=False,
        timeout=30
    )

    response.raise_for_status()

    response.encoding = "utf-8"

    html = response.text

    match = re.search(
        r"const DATA = (\[.*?\]);",
        html,
        re.DOTALL
    )

    if not match:
        raise Exception("DATA no encontrado")

    data_str = match.group(1)

    data = json5.loads(data_str)

    return data


def get_remote_estado():

    response = requests.get(
        ESTADO_URL,
        verify=False,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def save_snapshot(data):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO mando_snapshots (data)
        VALUES (%s)
        """,
        (json.dumps(data),)
    )

    conn.commit()

    cur.close()
    conn.close()


def main():

    projects = load_remote_dashboard()

    estado = get_remote_estado()

    for project in projects:

        project_id = project["id"]

        for idx, task in enumerate(project["tasks"]):

            key = f"{project_id}_{idx}"

            task["done"] = estado.get(key, False)

    snapshot = {
        "projects": projects
    }

    save_snapshot(snapshot)

    print("Snapshot estructurado guardado")

if __name__ == "__main__":

    main()