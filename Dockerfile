# Dockerfile (root of repo)

FROM python:3.11-slim

# xgboost потребує libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["streamlit", "run", "web/app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]