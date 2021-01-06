FROM python:3.9.1-buster

COPY Pipfile* /

RUN pip install pipenv 

RUN pipenv install

RUN pipenv install --system --deploy

