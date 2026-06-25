from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    database: str
    user: str
    password: str
    port: int


def get_database_settings() -> DatabaseSettings:
    """Carga configuracion de base de datos desde variables de entorno."""
    return DatabaseSettings(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "ia"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "Globallogic2026@"),
        port=int(os.getenv("DB_PORT", "3306")),
    )
