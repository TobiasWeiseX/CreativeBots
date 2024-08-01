FROM python:3.12

RUN apt-get update
RUN apt-get install -y firefox-esr
RUN apt-get install -y ffmpeg
RUN apt-get install -y espeak-ng

#RUN curl https://ollama.ai/install.sh | sh
#RUN ollama run llama2

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

ENTRYPOINT ["python3", "app.py"]
#ENTRYPOINT ["fastapi", "run", "main.py", "--port", "8000"]

