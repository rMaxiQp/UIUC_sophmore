FROM ubuntu:latest

LABEL "CLASS"="CS398"
LABEL "NETID"="zq2"

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev
  #&& pip3 install --upgrade pip

RUN pip3 install flask

RUN useradd -ms /bin/bash CS398

ADD info.txt /data/info.txt
ADD app.py /data/app.py

USER CS398

ENV SCHOOL uiuc

CMD python3 /data/app.py
