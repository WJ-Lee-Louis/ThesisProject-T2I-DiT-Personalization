FROM python:3.10.16-slim-bookworm
ENV PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface
WORKDIR /workspace
COPY freecus/requirements*.txt /tmp/freecus/
COPY freecus/constraints.txt /tmp/freecus/constraints.txt
RUN pip install --no-cache-dir -r /tmp/freecus/requirements-gpu.txt
COPY freecus /workspace/freecus
ENTRYPOINT ["python", "-m", "freecus.run"]
