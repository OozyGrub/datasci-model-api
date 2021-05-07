FROM python:3.7

WORKDIR /app


# Install dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt /app
RUN pip install -r requirements.txt

# App
COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]