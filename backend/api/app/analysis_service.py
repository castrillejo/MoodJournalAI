from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

# Intentamos usar psycopg (v3) y si no, psycopg2 (v2)
try:
    import psycopg  # type: ignore
    _PSYCOPG_V3 = True
except Exception:
    psycopg = None  # type: ignore
    _PSYCOPG_V3 = False

try:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore
    _PSYCOPG2 = True
except Exception:
    psycopg2 = None  # type: ignore
    _PSYCOPG2 = False


# =========================
# Config / Constantes
# =========================

LABEL_ORDER = ["joy", "sadness", "fear", "anger", "love", "surprise"]

# Para búsqueda "sin acentos" sin depender de extensiones:
# OJO: translate requiere mismas longitudes.
_TRANSLATE_FROM = "áéíóúüñÁÉÍÓÚÜÑ"
_TRANSLATE_TO   = "aeiouunAEIOUUN"


class NotFoundError(Exception):
    """Recurso no encontrado (usuario inexistente, etc.)."""


class DatabaseDriverError(Exception):
    """No hay driver de Postgres instalado."""


def _get_database_url() -> str:
    """
    Preferimos DATABASE_URL si existe.
    Si no, usamos variables sueltas típicas de docker-compose.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "moodjournal")
    user = os.getenv("POSTGRES_USER", "admin")
    pwd = os.getenv("POSTGRES_PASSWORD", "admin")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


def _connect():
    """
    Devuelve una conexión a Postgres (context manager compatible).
    """
    dsn = _get_database_url()

    if _PSYCOPG_V3 and psycopg is not None:
        # psycopg v3
        return psycopg.connect(dsn)
    if _PSYCOPG2 and psycopg2 is not None:
        # psycopg2
        return psycopg2.connect(dsn)

    raise DatabaseDriverError(
        "No se encontró driver de Postgres. Instala uno:\n"
        "  pip install psycopg[binary]\n"
        "o\n"
        "  pip install psycopg2-binary"
    )


def _fetchall_dict(conn, query: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    """
    Ejecuta un SELECT y devuelve lista de dicts (compatible v3/v2).
    """
    if _PSYCOPG_V3 and psycopg is not None:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:  # type: ignore
            cur.execute(query, params)
            return list(cur.fetchall())

    # psycopg2
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:  # type: ignore
        cur.execute(query, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def _fetchone_dict(conn, query: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
    rows = _fetchall_dict(conn, query, params)
    return rows[0] if rows else None


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den not in (0, 0.0) else 0.0


def _days_span(first: Optional[date], last: Optional[date]) -> int:
    """
    span en días incluyendo extremos (min 1 si hay fechas).
    """
    if not first or not last:
        return 0
    span = (last - first).days + 1
    return max(span, 1)


def _normalize_emotion(x: Any) -> str:
    return str(x).strip().lower()


# =========================
# 1) SEARCH (autocomplete)
# =========================

def search_users(q: str, limit: int = 5) -> Dict[str, Any]:
    """
    Devuelve sugerencias para autocomplete. NO calcula stats.
    """
    q = (q or "").strip()
    limit = int(limit or 5)
    limit = max(1, min(limit, 20))  # límite razonable

    if len(q) == 0:
        return {"query": q, "results": []}

    # Ajusta aquí si tus columnas/tabla se llaman diferente:
    # usuarios (id_usuario, nombre, sexo, edad, ocupacion, personalidad, p_actividad)
    query = f"""
        SELECT
            id_usuario,
            nombre,
            sexo,
            edad,
            ocupacion,
            personalidad,
            p_actividad
        FROM usuarios
        WHERE translate(lower(nombre), %s, %s)
              LIKE '%%' || translate(lower(%s), %s, %s) || '%%'
        ORDER BY
            CASE
              WHEN translate(lower(nombre), %s, %s) = translate(lower(%s), %s, %s) THEN 0
              WHEN translate(lower(nombre), %s, %s) LIKE translate(lower(%s), %s, %s) || '%%' THEN 1
              ELSE 2
            END,
            length(nombre) ASC,
            nombre ASC
        LIMIT %s
    """

    params = (
        _TRANSLATE_FROM, _TRANSLATE_TO,
        q, _TRANSLATE_FROM, _TRANSLATE_TO,
        _TRANSLATE_FROM, _TRANSLATE_TO, q, _TRANSLATE_FROM, _TRANSLATE_TO,
        _TRANSLATE_FROM, _TRANSLATE_TO, q, _TRANSLATE_FROM, _TRANSLATE_TO,
        limit,
    )

    with _connect() as conn:
        rows = _fetchall_dict(conn, query, params)

    # Devuelve "listo para dropdown"
    results = [
        {
            "id_usuario": r.get("id_usuario"),
            "nombre": r.get("nombre"),
            "sexo": r.get("sexo"),
            "edad": r.get("edad"),
            "ocupacion": r.get("ocupacion"),
            "personalidad": r.get("personalidad"),
            "p_actividad": r.get("p_actividad"),
        }
        for r in rows
    ]

    return {"query": q, "results": results}


# =========================
# 2) STATS (análisis completo)
# =========================

def get_user_stats(user_id: str) -> Dict[str, Any]:
    """
    Devuelve user + stats + charts listos para pintar.
    Aquí sí hacemos TODO el análisis.
    """
    user_id = (user_id or "").strip()
    if not user_id:
        raise NotFoundError("user_id vacío")

    with _connect() as conn:
        # 1) Usuario
        user = _fetchone_dict(
            conn,
            """
            SELECT id_usuario, nombre, sexo, edad, ocupacion, personalidad, p_actividad
            FROM usuarios
            WHERE id_usuario = %s
            """,
            (user_id,),
        )
        if not user:
            raise NotFoundError(f"Usuario no encontrado: {user_id}")

        # 2) Agregados de actividad
        agg = _fetchone_dict(
            conn,
            """
            SELECT
                COUNT(*)::int AS n_entries,
                COUNT(DISTINCT (fecha::date))::int AS active_days,
                MIN(fecha::date) AS first_date,
                MAX(fecha::date) AS last_date
            FROM entradas_diario
            WHERE id_usuario = %s
            """,
            (user_id,),
        ) or {}

        n_entries = int(agg.get("n_entries") or 0)
        active_days = int(agg.get("active_days") or 0)
        first_date = agg.get("first_date")
        last_date = agg.get("last_date")

        span_days = _days_span(first_date, last_date)

        # 3) Conteo por emoción
        emo_rows = _fetchall_dict(
            conn,
            """
            SELECT emocion_principal, COUNT(*)::int AS cnt
            FROM entradas_diario
            WHERE id_usuario = %s
            GROUP BY emocion_principal
            """,
            (user_id,),
        )

    # Construcción de counts completos en orden fijo
    emotion_counts: Dict[str, int] = {lab: 0 for lab in LABEL_ORDER}
    extra_emotions: Dict[str, int] = {}

    for r in emo_rows:
        emo = _normalize_emotion(r.get("emocion_principal"))
        cnt = int(r.get("cnt") or 0)
        if emo in emotion_counts:
            emotion_counts[emo] += cnt
        else:
            # por si hay etiquetas raras; no las tiramos
            extra_emotions[emo] = extra_emotions.get(emo, 0) + cnt

    # Shares
    emotion_share: Dict[str, float] = {lab: 0.0 for lab in LABEL_ORDER}
    if n_entries > 0:
        for lab in LABEL_ORDER:
            emotion_share[lab] = float(emotion_counts[lab] / n_entries)

    # Top emotion
    if n_entries > 0:
        top_emotion = max(LABEL_ORDER, key=lambda lab: emotion_counts[lab])
        top_emotion_share = float(emotion_share[top_emotion])
    else:
        top_emotion = ""
        top_emotion_share = 0.0

    # Diversidad emocional (nº de emociones distintas con count>0)
    diversidad_emocional = int(sum(1 for lab in LABEL_ORDER if emotion_counts[lab] > 0))

    # Actividad
    if n_entries > 0:
        activity_rate = _safe_div(active_days, span_days if span_days > 0 else 1)
        entries_per_week = _safe_div(n_entries * 7.0, span_days if span_days > 0 else 1)
    else:
        activity_rate = 0.0
        entries_per_week = 0.0

    # Charts (listas ya listas para pintar) — orden descendente por valor
    charts_emotion_counts = [
        {"emotion": lab, "value": emotion_counts[lab]}
        for lab in LABEL_ORDER
        if emotion_counts[lab] > 0
    ]
    charts_emotion_counts.sort(key=lambda x: x["value"], reverse=True)

    charts_emotion_share = [
        {"emotion": lab, "value": emotion_share[lab]}
        for lab in LABEL_ORDER
        if emotion_share[lab] > 0
    ]
    charts_emotion_share.sort(key=lambda x: x["value"], reverse=True)

    stats = {
        "n_entries": n_entries,
        "activity_rate": float(activity_rate),
        "entries_per_week": float(entries_per_week),

        "emotion_counts": emotion_counts,
        "emotion_share": emotion_share,

        "top_emotion": top_emotion,
        "top_emotion_share": float(top_emotion_share),

        "diversidad_emocional": diversidad_emocional,
    }

    # Si quieres exponer emociones "extra" (por si existen), aquí:
    # stats["emotion_counts_extra"] = extra_emotions

    return {
        "user": user,
        "stats": stats,
        "charts": {
            "emotion_counts": charts_emotion_counts,
            "emotion_share": charts_emotion_share,
        },
        "meta": {
            "labels_order": LABEL_ORDER,
        },
    }
