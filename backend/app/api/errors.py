from fastapi import HTTPException


def api_error(status: int, code: str, message: str) -> HTTPException:
    """HTTPException whose body is the SPEC 7 shape {error, message} (see main.py's handler)."""
    return HTTPException(status_code=status, detail={"error": code, "message": message})
