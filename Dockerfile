# Образ застосунку. Сервіси висновування, ядро узгодження та REST API
# запускаються з нього різними командами (див. docker-compose.yml).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir --no-deps .

ENTRYPOINT ["python", "-m", "hybrid_ids"]
CMD ["serve"]
