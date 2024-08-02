FROM python:3.12

RUN apt-get update
RUN apt-get install -y firefox-esr
RUN apt-get install -y ffmpeg
RUN apt-get install -y espeak

#RUN curl https://ollama.ai/install.sh | sh
#RUN ollama run llama2
WORKDIR /code

COPY requirements.txt /code/requirements.txt
#RUN pip3 install --no-cache-dir --upgrade -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt


COPY . .

ENTRYPOINT ["python3", "/code/app.py"]
#ENTRYPOINT ["fastapi", "run", "main.py", "--port", "8000"]

