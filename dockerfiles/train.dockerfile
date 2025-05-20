FROM convlogic-base

WORKDIR /workspace/convlogic

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install --upgrade requests

COPY ./ ./
RUN pip install .

ENTRYPOINT ["python3.11", "src/train.py"]
CMD []
