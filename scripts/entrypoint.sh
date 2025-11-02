set -e

export AIRFLOW_HOME=/opt/airflow/airflow

echo "Initializing Airflow database"
airflow db init

echo "Creating admin user if missing"
airflow users create \
  --username "${AIRFLOW_USERNAME}" \
  --firstname "${AIRFLOW_FIRSTNAME}" \
  --lastname "${AIRFLOW_LASTNAME}" \
  --role "${AIRFLOW_ROLE}" \
  --email "${AIRFLOW_EMAIL}" \
  --password "${AIRFLOW_ADMIN_PASSWORD}" 

echo "Cleaning up stale webserver PID"
rm -f "$AIRFLOW_HOME/airflow-webserver.pid"

echo "Starting Airflow scheduler"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
airflow scheduler &

echo "Starting Airflow webserver"
exec airflow webserver
