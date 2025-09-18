# Use an official Python image as base
FROM python:3.12

# Set the working directory
WORKDIR /app

# 1) Install Python deps from your requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2) Copy source code
COPY src/ ./src

# 3) Copy scripts
COPY bin/ /app/.
RUN chmod +x /app/bin/*.sh

# Expose port
EXPOSE 8000

# Run the application
CMD ["/app/start.sh"]
