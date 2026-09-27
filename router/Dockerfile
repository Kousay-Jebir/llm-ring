FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY router/requirements.txt /app/router-requirements.txt
RUN pip install -r /app/router-requirements.txt

COPY router /app/router

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER 10001

EXPOSE 8000

CMD ["uvicorn", "--factory", "router.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
