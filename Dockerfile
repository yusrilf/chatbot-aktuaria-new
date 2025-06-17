FROM python:3.9-slim

# Set env
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose Flask (5000) and Streamlit (8501)
EXPOSE 5000
EXPOSE 8501

# Jalankan Streamlit & Flask
CMD ["bash", "-c", "streamlit run frontend/app.py & python3 -m app.main"]
