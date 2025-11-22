# 1. docker build -t bank-deposit-api# 1. light-weight Python 3.12 image
FROM python:3.12-slim

# 2. set working folder inside container
WORKDIR /app

# 3. copy dependency list first (Docker layer caching)
COPY requirements.txt .

# 4. install deps
RUN pip install --no-cache-dir -r requirements.txt

# 5. copy model + service code
COPY model.bin predict.py ./

# 6. expose port (standard for FastAPI)
EXPOSE 8000

# 7. default command = start uvicorn
# last line in Dockerfile
CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port 8000 2>&1 | tee /dev/stderr"]