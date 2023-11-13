from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import RedirectResponse

from config.settings import Settings

router = APIRouter(tags=["Default"])


@router.get("/")
def default_route(req: Request):
    settings: Settings = req.app.state.settings
    return RedirectResponse(f'{settings.prefix}/docs')
