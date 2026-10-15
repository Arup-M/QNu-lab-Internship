FROM python:3.10

# Set working directory
WORKDIR /app

# Avoid buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY . .

# Create shared volume
RUN mkdir -p /shared
VOLUME ["/shared"]

# Default entrypoint executes the file specified by ROLE (e.g., client.py, server.py, main.py)
CMD sh -c "python -u /app/${ROLE}.py"


