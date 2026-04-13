from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class ClassifyRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="WhatsApp message to classify")

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


class TaskResult(BaseModel):
    intent: Optional[str] = Field(None, description="TASK_ASSIGN / TASK_DONE / TASK_UPDATE / PROGRESS_NOTE / GENERAL_MESSAGE")
    assigneeName: Optional[str] = None
    project: Optional[str] = None
    title: Optional[str] = None
    deadline: Optional[str] = None
    priority: Optional[str] = None
    progressPercent: Optional[int] = None


class ClassifyResponse(BaseModel):
    id: str = Field(..., description="Unique request ID")
    model: str = Field(..., description="Model version used")
    message: str = Field(..., description="Original input message")
    result: Optional[TaskResult] = None
    raw_output: str = Field(..., description="Raw model output before parsing")
    parse_success: bool = Field(..., description="Whether the output was valid JSON")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class BatchRequest(BaseModel):
    messages: List[str] = Field(..., min_length=1, description="List of messages to classify")

    @field_validator("messages")
    @classmethod
    def check_batch_size(cls, v: list) -> list:
        if len(v) > 10:
            raise ValueError("Maximum batch size is 10 messages")
        if any(not m.strip() for m in v):
            raise ValueError("Messages cannot be empty")
        return [m.strip() for m in v]


class BatchItem(BaseModel):
    index: int
    message: str
    result: Optional[TaskResult] = None
    raw_output: str
    parse_success: bool
    latency_ms: float


class BatchResponse(BaseModel):
    id: str
    model: str
    total: int
    successful: int
    failed: int
    results: List[BatchItem]
    total_latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    base_model: str
    adapter_dir: str
    device: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    code: str
    request_id: Optional[str] = None


# ── OpenAI-compatible schemas ──────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="system | user | assistant")
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(150, ge=1, le=2048)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(150, ge=1, le=2048)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stop: Optional[List[str]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
