FROM daocloud.io/quantaxis/qabase:latest

COPY requirements.txt /requirements.txt
COPY entrypoint.sh /entrypoint.sh
COPY jupyter_notebook_config.py /root/.jupyter/

WORKDIR home

RUN wget https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
	&& tar xvf ta-lib-0.4.0-src.tar.gz \
	&& cd ta-lib \
	&& ./configure --prefix=/usr \
	&& make \
	&& make install \
  && cd .. \
	&& rm -rf ta-lib \
	&& rm ta-lib-0.4.0-src.tar.gz \
  && pip install -r /requirements.txt \
  && pip install prompt-toolkit \
  && jupyter nbextension enable --py widgetsnbextension \
  && jupyter serverextension enable --py jupyterlab
RUN chmod +x /entrypoint.sh
EXPOSE 8888 

COPY runpy.sh /root/
RUN chmod +x /QUANTAXIS/QUANTAXIS/QAUtil/QASetting.py

RUN chmod +x /root/runpy.sh
CMD ["bash", "/root/runpy.sh"]
