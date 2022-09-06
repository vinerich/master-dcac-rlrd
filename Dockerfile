FROM python:3.9

RUN apt-get -y update \
    && apt-get -y install swig

ENV CODE_DIR /root/code
ENV VENV /root/venv
COPY requirements.txt /tmp/

# # RUN pip install --upgrade pip

RUN \
    mkdir -p ${CODE_DIR}/rl_zoo && \
    pip uninstall -y stable-baselines3 && \
    pip install -r /tmp/requirements.txt && \
    rm -rf $HOME/.cache/pip

RUN pip install git+https://github.com/vinerich/zinc-coating-gym-env.git

ENV PWD=/app
RUN mkdir /app
WORKDIR $PWD

RUN mkdir /logs

COPY . .

RUN chmod +x ./train.sh

ENV PATH=$VENV/bin:$PATH

CMD ["/app/train.sh"]