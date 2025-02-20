FROM python:3.9.16-slim-buster
#
#WORKDIR app
#COPY ./ ./

COPY . /app
WORKDIR /app

ENV PYTHONPATH=/app

COPY pyproject.toml .
COPY pdm.lock .

RUN pip install -U pip setuptools wheel
RUN pip install pdm
RUN pdm install --prod --no-lock --no-editable

ENTRYPOINT ["pdm", "run", "src/server.py"]
