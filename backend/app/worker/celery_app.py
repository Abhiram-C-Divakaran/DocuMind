from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init

from backend.app.config import settings
from backend.app.logger import logger

celery_app = Celery(
    "documind_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["backend.app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=settings.CELERY_WORKER_MAX_TASKS_PER_CHILD,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_always_eager=settings.CELERY_TASK_ALWAYS_EAGER,
    task_eager_propagates=False,
    broker_connection_retry_on_startup=True,
    beat_schedule={
        "recover-stale-jobs-and-outbox-every-minute": {
            "task": "documind.recover_stale_jobs_and_outbox",
            "schedule": crontab(minute="*"),
        },
    },
)


@worker_process_init.connect
def _on_worker_process_init(**kwargs) -> None:
    try:
        from ml.extractors.layoutlmv3_extractor import preload_worker_model_if_ready

        preload_worker_model_if_ready()
    except Exception as exc:
        logger.warning(
            "Worker startup LayoutLMv3 preload failed; inference will initialize lazily",
            extra={"event": "worker_preload_warning", "error_category": type(exc).__name__},
        )

