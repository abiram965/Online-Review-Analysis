# Use an official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory inside container
WORKDIR /app

# Copy project files into the container
COPY . /app/

# # Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
# Expose the port your Flask app runs on
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
