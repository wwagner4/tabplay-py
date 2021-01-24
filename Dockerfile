FROM python:3.9.1-buster

RUN python -m pip install --upgrade pip

COPY Pipfile* /

RUN pip install pipenv 

RUN pipenv install --system --deploy
