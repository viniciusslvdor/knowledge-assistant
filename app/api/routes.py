from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse
from app.core.config import Settings
from app.services.rag_service import RAGService

router = APIRouter()


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


ServiceDep = Annotated[RAGService, Depends(get_rag_service)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]


@router.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health(service: ServiceDep):
    result = service.health()
    status_code = 200 if result.status == "ok" else 503
    return JSONResponse(status_code=status_code, content=result.model_dump())


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        502: {"model": ErrorResponse, "description": "The LLM returned invalid output"},
        503: {"model": ErrorResponse, "description": "A dependency is unavailable"},
        504: {"model": ErrorResponse, "description": "The LLM timed out"},
    },
)
def ask(payload: AskRequest, service: ServiceDep, settings: SettingsDep) -> AskResponse:
    # Sync handler: FastAPI runs it in a worker thread, so blocking model calls
    # do not block the event loop.
    if len(payload.question) > settings.max_question_length:
        raise RequestValidationError(
            [
                {
                    "type": "string_too_long",
                    "loc": ("body", "question"),
                    "msg": f"question must be at most {settings.max_question_length} characters",
                    "input": None,
                    "ctx": {"max_length": settings.max_question_length},
                }
            ]
        )
    return service.ask(payload.question)
