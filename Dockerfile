FROM --platform=linux/amd64 pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
LABEL authors="MumuAktar"

RUN apt-get update -y

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY tools tools/
COPY checkpoints checkpoints/
COPY main.py .

CMD ["python", "main.py", "-i", "/input", "-o", "/output"]

