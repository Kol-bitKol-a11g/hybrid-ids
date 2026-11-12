# Сенсор потокових ознак: Python-реалізація CICFlowMeter.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libpcap0.8 tcpdump \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir cicflowmeter

ENTRYPOINT ["cicflowmeter"]
