from core.db import get_connection


conn = get_connection()

print("PostgreSQL connected")

conn.close()

print("Connection closed")