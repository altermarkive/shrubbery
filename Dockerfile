FROM ghcr.io/marek-burza/utilities:latest

ENV UV_NO_CACHE=1
# TensorRT is quite noisy
ENV PYTHONWARNINGS="ignore::SyntaxWarning"
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    /bin/uv sync --active --frozen --no-dev --no-install-project
COPY --chown=$USER:$USER . /app/shrubbery
RUN /bin/uv pip install --python $VIRTUAL_ENV "/app/shrubbery[dev]" && \
    rm -rf /app/shrubbery
ENV NUMERAI_MODEL_PATH=/app/model.pkl.zip
COPY src/shrubbery/example.py /app/model.py
ENTRYPOINT ["/app/venv/bin/python", "/app/model.py"]
