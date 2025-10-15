FROM convlogic-base

WORKDIR /workspace/convlogic

ENV FORCE_CUDA=1

COPY ./ ./
RUN pip install .

ENTRYPOINT ["python3", "src/train.py"]
CMD []
