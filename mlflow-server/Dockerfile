FROM python:3.11-alpine

RUN apk add --no-cache --virtual .build-deps \
        gcc g++ musl-dev linux-headers \
        libffi-dev openssl-dev \
        make cmake \
    && pip install --no-cache-dir mlflow \
    && apk del .build-deps

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--allowed-hosts", "*", \
     "--cors-allowed-origins", "*"]

# podman build --no-cache -t registry.accordions.co.kr/mlflow-server:ysmoon .
# podman push registry.accordions.co.kr/mlflow-server:ysmoon