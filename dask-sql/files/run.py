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

    wait(df)

    c.create_table("nyc-taxi", df)

    c.sql("SELECT 1 + 1").compute()

    # Finally, spin up the dask-sql server
    run_server(context=c, client=client)