FROM python:3.12

RUN apt-get update
RUN apt-get install -y firefox-esr
RUN apt-get install -y ffmpeg
RUN apt-get install -y espeak
RUN apt-get install -y flite

#RUN curl https://ollama.ai/install.sh | sh
#RUN ollama run llama2
WORKDIR /code

COPY requirements.txt /code/requirements.txt
#RUN pip3 install --no-cache-dir --upgrade -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 freeze > current_requirements.txt

COPY . .

ENTRYPOINT ["python3", "/code/app.py"]
#ENTRYPOINT ["fastapi", "run", "main.py", "--port", "8000"]

#gunicorn -w 4 -b 0.0.0.0 'hello:create_app()'
#ENTRYPOINT ["gunicorn", "-w", "1", "-b", "0.0.0.0", "app:create_app()"]

#ENTRYPOINT ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:create_app()"]
#gunicorn app:app --worker-class eventlet -w 1 --bind 0.0.0.0:5000 --reload

