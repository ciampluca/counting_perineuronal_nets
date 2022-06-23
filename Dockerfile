FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt update && apt install -y build-essential wget unzip
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install jupyter ipython==7.23.1
RUN pip install jupyter_contrib_nbextensions nbconvert && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable collapsible_headings/main
