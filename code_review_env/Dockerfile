FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code_review_env/ ./code_review_env/
COPY server/ ./server/
COPY gradio_demo.py .
COPY launch.py .
COPY openenv.yaml .
COPY setup.py .

RUN pip install -e .

EXPOSE 7860

CMD ["python", "launch.py"]
