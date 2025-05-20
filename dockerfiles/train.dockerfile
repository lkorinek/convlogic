FROM convlogic-base

WORKDIR /workspace/convlogic

ENV FORCE_CUDA=1

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install --upgrade requests

COPY ./ ./
RUN pip install -v . --no-build-isolation

ENTRYPOINT ["python3", "src/train.py"]
CMD []
