# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Expose the port used by Hugging Face Spaces
EXPOSE 7860

# Run FastAPI app (main.py should contain app = FastAPI())
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
