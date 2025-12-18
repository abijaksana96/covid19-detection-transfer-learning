FROM python:3.9

WORKDIR /code

# Copy requirements file
COPY ./requirements.txt /code/requirements.txt 

# Install dependencies
RUN pip install --no-cache-dir --default-timeout=100 -i https://pypi.org/simple -r requirements.txt

# Copy the rest of the application code
COPY ./app /code/app

EXPOSE 8000

CMD [ "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000" ]