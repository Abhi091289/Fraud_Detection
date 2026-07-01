FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.14.0
WORKDIR /mlflow
EXPOSE 5000
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlartifacts", \
     "--workers", "1", \
     "--allowed-hosts", "mlflow,localhost,127.0.0.1,api,fg_api,fg_mlflow,0.0.0.0,::1"]
