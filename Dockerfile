FROM python:3.8

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libjpeg-dev \
    libfreetype6-dev \
    libpng-dev

RUN pip install --no-cache-dir \
    tensorflow \
    tensorflow-datasets \
    matplotlib \
    numpy

COPY ./src /app