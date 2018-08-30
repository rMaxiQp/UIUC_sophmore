FROM tiangolo/uwsgi-nginx-flask:python3.6

RUN pip3 install bs4 && pip3 install requests

COPY ./templates /data/templates
ADD app.py /data

CMD python3 /data/app.py

