import json
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

from api.config import settings
from api.inference import model_instance
from api.schemas import (
    BatchItem,
    BatchRequest,
    BatchResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    ClassifyRequest,
    ClassifyResponse,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelList,
    TaskResult,
    UsageInfo,
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("taskmind")

_start_time = time.time()
_request_counts: dict = defaultdict(int)
_rate_window: dict = defaultdict(float)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _check_rate_limit(client_id: str) -> bool:
    now = time.time()
    window_start = _rate_window.get(client_id, 0)
    if now - window_start > 60:
        _rate_window[client_id] = now
        _request_counts[client_id] = 1
        return True
    _request_counts[client_id] += 1
    return _request_counts[client_id] <= settings.RATE_LIMIT_PER_MINUTE


def _get_client_id(request: Request, api_key: Optional[str]) -> str:
    return api_key or request.client.host or "anonymous"


def _verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    if not settings.REQUIRE_AUTH:
        return api_key
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid or missing API key", "code": "UNAUTHORIZED"},
        )
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TaskMind API starting up ...")
    model_instance.load()
    logger.info("Model loaded. Ready to serve.")
    yield
    logger.info("TaskMind API shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "Production API for TaskMind — a fine-tuned TinyLlama model that extracts "
        "structured task data from team WhatsApp messages."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{latency:.1f}"
    logger.info(
        "method=%s path=%s status=%d latency_ms=%.1f request_id=%s",
        request.method, request.url.path, response.status_code, latency, request_id,
    )
    return response


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health(request: Request):
    return HealthResponse(
        status="ok" if model_instance.is_loaded else "degraded",
        model_loaded=model_instance.is_loaded,
        model_version=settings.MODEL_VERSION,
        base_model=settings.BASE_MODEL,
        adapter_dir=settings.ADAPTER_DIR,
        device=model_instance.device,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/metrics", tags=["Ops"])
def metrics():
    return {
        "uptime_seconds": round(time.time() - _start_time, 1),
        "model_loaded": model_instance.is_loaded,
        "rate_limit_per_minute": settings.RATE_LIMIT_PER_MINUTE,
        "active_clients": len(_request_counts),
    }


@app.post("/v1/classify", response_model=ClassifyResponse, tags=["Inference"])
def classify(
    req: ClassifyRequest,
    request: Request,
    api_key: Optional[str] = Security(_verify_api_key),
):
    client_id = _get_client_id(request, api_key)
    if not _check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": "Rate limit exceeded", "code": "RATE_LIMITED"},
        )

    if not model_instance.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "Model not ready", "code": "MODEL_UNAVAILABLE"},
        )

    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:12])
    t0 = time.perf_counter()

    try:
        raw, parsed, success = model_instance.classify(req.message)
    except Exception as exc:
        logger.error("Inference error request_id=%s: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Inference failed", "code": "INFERENCE_ERROR"},
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    result = TaskResult(**parsed) if success and parsed else None

    return ClassifyResponse(
        id=f"req_{request_id}",
        model=settings.MODEL_VERSION,
        message=req.message,
        result=result,
        raw_output=raw,
        parse_success=success,
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/v1/batch", response_model=BatchResponse, tags=["Inference"])
def batch_classify(
    req: BatchRequest,
    request: Request,
    api_key: Optional[str] = Security(_verify_api_key),
):
    if len(req.messages) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": f"Batch size exceeds limit of {settings.MAX_BATCH_SIZE}", "code": "BATCH_TOO_LARGE"},
        )

    client_id = _get_client_id(request, api_key)
    if not _check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": "Rate limit exceeded", "code": "RATE_LIMITED"},
        )

    if not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail="Model not ready")

    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:12])
    batch_start = time.perf_counter()
    results = []
    successful = 0

    for i, message in enumerate(req.messages):
        t0 = time.perf_counter()
        try:
            raw, parsed, success = model_instance.classify(message)
            if success:
                successful += 1
        except Exception as exc:
            logger.warning("Batch item %d failed: %s", i, exc)
            raw, parsed, success = str(exc), None, False

        latency_ms = (time.perf_counter() - t0) * 1000
        result = TaskResult(**parsed) if success and parsed else None
        results.append(BatchItem(
            index=i,
            message=message,
            result=result,
            raw_output=raw,
            parse_success=success,
            latency_ms=round(latency_ms, 2),
        ))

    total_latency = (time.perf_counter() - batch_start) * 1000

    return BatchResponse(
        id=f"batch_{request_id}",
        model=settings.MODEL_VERSION,
        total=len(req.messages),
        successful=successful,
        failed=len(req.messages) - successful,
        results=results,
        total_latency_ms=round(total_latency, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/v1/models", response_model=ModelList, tags=["OpenAI-Compatible"])
def list_models():
    return ModelList(data=[
        ModelInfo(
            id=settings.MODEL_VERSION,
            created=int(_start_time),
            owned_by="taskmind",
        )
    ])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["OpenAI-Compatible"])
def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    api_key: Optional[str] = Security(_verify_api_key),
):
    client_id = _get_client_id(request, api_key)
    if not _check_rate_limit(client_id):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail={"error": "Rate limit exceeded", "code": "RATE_LIMITED"})
    if not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail={"error": "Model not ready", "code": "MODEL_UNAVAILABLE"})
    if req.stream:
        raise HTTPException(status_code=400, detail={"error": "Streaming not supported", "code": "STREAM_UNSUPPORTED"})

    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:12])
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        text, prompt_tokens, completion_tokens = model_instance.chat_complete(
            messages,
            max_new_tokens=req.max_tokens or settings.MAX_NEW_TOKENS,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 1.0,
        )
    except Exception as exc:
        logger.error("chat_complete error request_id=%s: %s", request_id, exc)
        raise HTTPException(status_code=500, detail={"error": "Inference failed", "code": "INFERENCE_ERROR"})

    return ChatCompletionResponse(
        id=f"chatcmpl_{request_id}",
        created=int(time.time()),
        model=settings.MODEL_VERSION,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.post("/v1/completions", response_model=CompletionResponse, tags=["OpenAI-Compatible"])
def completions(
    req: CompletionRequest,
    request: Request,
    api_key: Optional[str] = Security(_verify_api_key),
):
    client_id = _get_client_id(request, api_key)
    if not _check_rate_limit(client_id):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail={"error": "Rate limit exceeded", "code": "RATE_LIMITED"})
    if not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail={"error": "Model not ready", "code": "MODEL_UNAVAILABLE"})

    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:12])

    try:
        text, prompt_tokens, completion_tokens = model_instance.complete(
            req.prompt,
            max_new_tokens=req.max_tokens or settings.MAX_NEW_TOKENS,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 1.0,
        )
    except Exception as exc:
        logger.error("complete error request_id=%s: %s", request_id, exc)
        raise HTTPException(status_code=500, detail={"error": "Inference failed", "code": "INFERENCE_ERROR"})

    return CompletionResponse(
        id=f"cmpl_{request_id}",
        created=int(time.time()),
        model=settings.MODEL_VERSION,
        choices=[CompletionChoice(text=text, index=0, finish_reason="stop")],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", None)
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content={**detail, "request_id": request_id},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(detail), "code": "ERROR", "request_id": request_id},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.HOST, port=settings.PORT, reload=False)
