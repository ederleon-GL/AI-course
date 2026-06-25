from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
import pymysql.cursors

from src.API.db import get_connection

app = FastAPI(
    title="WorldCup API",
    description="API para consultar partidos del mundial desde MySQL",
    version="1.0.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "WorldCup API activa",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                cursor.fetchone()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB no disponible: {exc}") from exc

    return {"status": "ok"}


@app.get("/matches")
def get_matches(
    limit: int = Query(default=10, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    query = """
    SELECT *
    FROM partidos
    ORDER BY year, datetime
    LIMIT %s OFFSET %s
    """

    with get_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(query, (limit, offset))
            return list(cursor.fetchall())


@app.get("/matches/year/{year}")
def get_matches_by_year(year: int) -> list[dict[str, Any]]:
    query = """
    SELECT *
    FROM partidos
    WHERE year = %s
    ORDER BY datetime
    """

    with get_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(query, (year,))
            rows = list(cursor.fetchall())

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron partidos para el anio {year}",
        )
    return rows


@app.get("/stats/matches-by-year")
def matches_by_year() -> list[dict[str, Any]]:
    query = """
    SELECT
        year,
        COUNT(*) AS matches
    FROM partidos
    GROUP BY year
    ORDER BY year
    """

    with get_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(query)
            return list(cursor.fetchall())
