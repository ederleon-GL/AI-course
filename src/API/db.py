from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pymysql
from pymysql.connections import Connection

from src.API.config import get_database_settings


@contextmanager
def get_connection() -> Iterator[Connection]:
    """Abre y cierra una conexion MySQL por uso."""
    settings = get_database_settings()
    conn = pymysql.connect(
        host=settings.host,
        database=settings.database,
        user=settings.user,
        password=settings.password,
        port=settings.port,
    )
    try:
        yield conn
    finally:
        conn.close()
