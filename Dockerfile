ARG ENDGRAIN_DEPS_IMAGE=python:3.11-slim
FROM ${ENDGRAIN_DEPS_IMAGE} AS python_deps
USER root
WORKDIR /build/endgrain
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 ENDGRAIN_DATA=/var/lib/endgrain
WORKDIR /opt/endgrain
COPY --from=python_deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY endgrain ./endgrain
COPY static ./static
COPY app.py .
RUN useradd -U -u 11480 endgrain && install -d -o endgrain -g endgrain /var/lib/endgrain
USER endgrain
EXPOSE 8000
CMD ["python", "app.py"]
