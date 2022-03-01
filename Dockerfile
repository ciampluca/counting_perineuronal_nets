FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt update && apt install -y build-essential
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install jupyter ipython==7.23.1
RUN pip install jupyter_contrib_nbextensions "nbconvert<6" && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable collapsible_headings/main

RUN apt install -y wget unzip
