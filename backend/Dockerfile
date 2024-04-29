FROM python:3.12

#RUN curl https://ollama.ai/install.sh | sh
#RUN ollama run llama2

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

ENTRYPOINT ["python3", "app.py"]

