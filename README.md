
# dask-sql k8s example deployment

This repository contains some example code to deploy `dask-sql`, a Dask cluster and
Apache Hue as BI tool on a k8s cluster.

## Requirements

You need to have a k8s cluster. You can either run a development k8s cluster
locally e.g. via [kind](https://kind.sigs.k8s.io/docs/user/quick-start/)
or [minikube](https://minikube.sigs.k8s.io/docs/start/)
or deploy a cluster on one of the public cloud providers.

After that, make sure you have `kubectl` and `helm` installed
and you can access your cluster.

## Deployment

First, make sure the file `dask-sql/values.yaml` contains
the correct number of workers you want to have and add additional conda packages to install.

Then, call

    helm dependency update dask-sql
    helm upgrade --cleanup-on-fail --install dask-sql dask-sql

After the deployment has finished and all pods are running, do a port-forwarding

    kubectl port-forward svc/hue 8888:8888

and access "http://localhost:8888". You should be able to see the "nyc-taxi" table in the schema called "schema"
in the presto tab.
Please note, that the first access to the server triggers some initialization, which might take a couple
of seconds.

If the `dask-sql` pod is constantly restarting and not getting into running state and the log is stuck at

    Solving environment: ...working... failed with initial frozen solve. Retrying with flexible solve.

try to increase the number `dask_sql.probeDelay` in `dask-sql/values.yaml`.

Now, you can query the data.
For example, try the following query:

```sql
SELECT
    FLOOR(trip_distance / 5) * 5 AS "distance",
    AVG(tip_amount) AS "given tip",
    AVG(predict_price(total_amount, trip_distance, passenger_count)) AS "predicted tip"
FROM "nyc-taxi"
WHERE
    trip_distance > 0 AND trip_distance < 50
GROUP BY
    FLOOR(trip_distance / 5) * 5
```

## How does it work?

The helm chart installs three components:

### Dask cluster

The basis for `dask-sql` is a Dask cluster. Dask already comes with a nice helm chart,
which has many configuration parameters.
We use it via a dependency and just change the number of workers
and the installed packages.

### Apache Hue

For accessing the SQL server, we use the Apache Hue BI tool.
It consists of the webservice and a MySQL database for the settings, which are
deployed using the manifest files in `dask-sql/templates/hue/`.

### `dask-sql`

Finally we can start the `dask-sql` container with a custom startup file, which
looks like this:

```python
import tempfile
import requests
from dask.distributed import Client, wait
import dask.dataframe as dd
import xgboost
import numpy as np
import dask_xgboost


if __name__ == "__main__":
    # Create a dask client
    client = Client("dask-sql-scheduler:8786")
    print("Dashboard:", client.dashboard_link)

    # Load model and register predict function
    bst = xgboost.core.Booster()

    with tempfile.NamedTemporaryFile() as f:
        r = requests.get("https://storage.googleapis.com/dask-sql-data/model.xgboost")
        r.raise_for_status()
        f.write(r.content)
        f.flush()
        bst.load_model(f.name)

    # Our custom function for tip-prediction
    # using the already loaded xgboost model
    def predict_price(total_amount, trip_distance, passenger_count):
        # Create a dataframe out of the three columns
        # and pass it to dask-xgboost, to predict
        # distributed
        X = dd.concat([total_amount, trip_distance, passenger_count],
                        axis=1).astype("float64")
        return dask_xgboost.predict(client, bst, X)

    # Create a context
    from dask_sql import Context, run_server
    c = Context()

    c.register_function(predict_price, "predict_price",
                        [("total_amount", np.float64),
                            ("trip_distance", np.float64),
                            ("passenger_count", np.float64)],
                        np.float64)

    # Load the data from S3
    df = dd.read_csv("s3://nyc-tlc/trip data/yellow_tripdata_2019-01.csv",
        dtype={
            "payment_type": "UInt8",
            "VendorID": "UInt8",
            "passenger_count": "UInt8",
            "RatecodeIDq": "UInt8",
        },
        storage_options={"anon": True}
    ).persist()

    c.create_table("nyc-taxi", df)

    # Finally, spin up the dask-sql server
    run_server(context=c, client=client)
```

If you want to edit the startup file, you need to change `dask-sql/files/run.py`.
