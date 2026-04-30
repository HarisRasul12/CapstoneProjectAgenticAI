FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY .streamlit ./.streamlit
COPY src ./src
COPY evals ./evals
COPY tests ./tests
COPY streamlit_app.py ./

RUN uv sync --frozen --no-dev

ENV PORT=8080
EXPOSE 8080

CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.headless", "true"]
