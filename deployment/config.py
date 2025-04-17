from pydantic import BaseSettings

class CeleryConfig(BaseSettings):
    """Celery configuration settings."""
    REDIS_URL: str = "redis://redis:6379/0"
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    WORKER_CONCURRENCY: int = 2
    TASK_TIME_LIMIT: int = 3600
    RESULT_EXPIRES: int = 3600

    class Config:
        env_file = "deployment/.env" 