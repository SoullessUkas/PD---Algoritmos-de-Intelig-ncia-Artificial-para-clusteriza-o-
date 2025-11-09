FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# Dependências do sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar app e dados de exemplo
COPY streamlit_app.py ./app.py
COPY Country-data.csv ./Country-data.csv
COPY data-dictionary.csv ./data-dictionary.csv

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=${PORT}", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--browser.gatherUsageStats=false"]