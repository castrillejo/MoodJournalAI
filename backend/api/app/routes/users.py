from fastapi import APIRouter, HTTPException, Query

from ..analysis_service import (
    search_users,
    get_user_stats,
    NotFoundError,
    DatabaseDriverError,
)

router = APIRouter()


@router.get("/users/search")
def users_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
):
    """
    Autocomplete: devuelve máximo `limit` usuarios similares por nombre.
    NO calcula stats aquí.
    """
    try:
        return search_users(q=q, limit=limit)
    except DatabaseDriverError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/stats")
def users_stats(user_id: str):
    """
    Devuelve el análisis completo del usuario:
    user + stats + charts + meta
    """
    try:
        return get_user_stats(user_id=user_id)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DatabaseDriverError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
