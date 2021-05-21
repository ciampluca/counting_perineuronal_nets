FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt update && apt install -y build-essential
COPY requirements.txt .
RUN pip install -r requirements.txt
# RUN pip install jupyter
