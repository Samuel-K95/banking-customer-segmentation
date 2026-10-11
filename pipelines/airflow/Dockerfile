FROM apache/airflow:3.3.2

# Only the libraries; the project code is mounted at /opt/project (see docker-compose.yaml),
# so code changes need no rebuild. Pinning apache-airflow stops pip from up/downgrading it.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt
