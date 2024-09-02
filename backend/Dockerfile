#FROM python:3.12
FROM ubuntu

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip


#on debian:
#RUN apt-get install -y firefox-esr
RUN apt-get install -y firefox

RUN apt-get install -y ffmpeg
RUN apt-get install -y espeak
RUN apt-get install -y flite

#COPY "amdgpu-install_6.1.60103-1_all.deb" "amdgpu-install_6.1.60103-1_all.deb"
#RUN dpkg -i "amdgpu-install_6.1.60103-1_all.deb"
#RUN amdgpu-install -y
#RUN apt-get -y install rocm-device-libs


#install "apt-add-repository" command:
RUN apt-get -y install software-properties-common dirmngr apt-transport-https lsb-release ca-certificates
RUN apt-add-repository -r ppa:graphics-drivers/ppa
RUN add-apt-repository ppa:oibaf/graphics-drivers


#RUN apt install hipsolver rocm-gdb -y

#RUN apt-get -y install wget
#RUN apt-get update -y --allow-unauthenticated
#RUN wget https://repo.radeon.com/amdgpu-install/6.1.1/ubuntu/jammy/amdgpu-install_6.1.60101-1_all.deb
#RUN apt-get install ./amdgpu-install_6.1.60101-1_all.deb -y
#RUN amdgpu-install --usecase=rocm


#RUN wget https://repo.radeon.com/amdgpu-install/6.1.1/ubuntu/jammy/amdgpu-install_6.1.60101-1_all.deb
#RUN apt install ./amdgpu-install_6.1.60101-1_all.deb  -y
#RUN amdgpu-install --usecase=graphics,rocm -y
#RUN usermod -a -G render,video $LOGNAME


#RUN wget https://repo.radeon.com/amdgpu-install/6.1.1/ubuntu/jammy/amdgpu-install_6.1.60101-1_all.deb
#RUN apt install ./amdgpu-install_6.1.60101-1_all.deb -y
#RUN amdgpu-install --usecase=graphics,rocm -y


#RUN echo "deb http://deb.debian.org/debian/ bookworm main contrib non-free-firmware" > /etc/apt/sources.list
#RUN echo "deb http://deb.debian.org/debian/ bookworm main contrib non-free" > /etc/apt/sources.list

#RUN apt-add-repository contrib
#RUN apt-add-repository non-free
#RUN apt update -y

#RUN apt install nvidia-driver -y
#RUN apt-get install firmware-amd-graphics libgl1-mesa-dri libglx-mesa0 mesa-vulkan-drivers xserver-xorg-video-all -y

RUN apt-get update -y --allow-unauthenticated
RUN apt-get upgrade -y --allow-unauthenticated
RUN apt-get autoremove -y
RUN apt-get autoclean -y

#RUN curl https://ollama.ai/install.sh | sh
#RUN ollama run llama2
WORKDIR /code

COPY requirements.txt /code/requirements.txt
#RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

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


