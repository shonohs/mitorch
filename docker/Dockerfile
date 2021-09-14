FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends python3.9 python3.9-venv && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m venv /venv
ENV PATH=/venv/bin/:$PATH

RUN pip install --no-cache torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -U wheel && pip install --no-cache mitorch~=0.1.1 mitorch-models~=0.0.6

ENV MITORCH_DATABASE_URL=''
ENV MITORCH_STORAGE_URL=''

CMD ["miagent", "--data", "/datasets"]
