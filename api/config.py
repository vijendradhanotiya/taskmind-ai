import os


class Settings:
    APP_NAME: str = "TaskMind API"
    VERSION: str = "2.0.0"
    MODEL_VERSION: str = "taskmind-1.1b-lora-v2"

    BASE_MODEL: str = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ADAPTER_DIR: str = os.environ.get("ADAPTER_DIR", "out/taskmind_lora_r2")

    API_KEY: str = os.environ.get("TASKMIND_API_KEY", "")
    REQUIRE_AUTH: bool = os.environ.get("REQUIRE_AUTH", "false").lower() == "true"

    MAX_BATCH_SIZE: int = int(os.environ.get("MAX_BATCH_SIZE", "10"))
    MAX_NEW_TOKENS: int = int(os.environ.get("MAX_NEW_TOKENS", "150"))

    RATE_LIMIT_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))

    CORS_ORIGINS: list = os.environ.get("CORS_ORIGINS", "*").split(",")

    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    LOG_JSON: bool = os.environ.get("LOG_JSON", "true").lower() == "true"

    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("PORT", "8001"))


settings = Settings()
