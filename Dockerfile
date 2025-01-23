FROM python:3.8

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./api.py /code/api.py

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
