
FROM ubuntu
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y firefox curl ffmpeg espeak flite


#install "apt-add-repository" command:
RUN apt-get -y install software-properties-common dirmngr apt-transport-https lsb-release ca-certificates
RUN add-apt-repository ppa:graphics-drivers/ppa -y

#amd?
RUN add-apt-repository ppa:oibaf/graphics-drivers
RUN apt install -y ubuntu-drivers-common
RUN ubuntu-drivers autoinstall

RUN apt-get update -y --allow-unauthenticated
RUN apt-get upgrade -y --allow-unauthenticated
RUN apt-get autoremove -y
RUN apt-get autoclean -y

WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages
RUN pip3 freeze > current_requirements.txt

COPY . .

ENTRYPOINT ["python3", "/code/app.py"]

#gunicorn -w 4 -b 0.0.0.0 'hello:create_app()'
#ENTRYPOINT ["gunicorn", "-w", "1", "-b", "0.0.0.0", "app:create_app()"]
#ENTRYPOINT ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:create_app()"]
#gunicorn app:app --worker-class eventlet -w 1 --bind 0.0.0.0:5000 --reload


#ENTRYPOINT ["fastapi", "run", "main.py", "--port", "8000"]

#ENTRYPOINT ["uvicorn", "main:app", "--port", "8000", "--host", "0.0.0.0"]
