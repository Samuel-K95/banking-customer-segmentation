FROM apache/airflow:3.3.2

# Only the libraries; the project code is mounted at /opt/project (see docker-compose.yaml),
# so code changes need no rebuild. Pinning apache-airflow stops pip from up/downgrading it.
# After pip, fix the airflow console-script shebang: it otherwise points at /usr/python/bin/python3.13
# which cannot see packages installed under ~/.local (breaks entrypoint + healthchecks).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt \
  && sed -i '1c#!/home/airflow/.local/bin/python' /home/airflow/.local/bin/airflow
