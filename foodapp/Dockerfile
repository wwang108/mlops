FROM python:3.10.10-slim

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt 

COPY ./serving ./serving

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "80"]