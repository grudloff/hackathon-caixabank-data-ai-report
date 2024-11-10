import pandas as pd
import os
import pickle
import logging
import json
from src.data.data_functions import get_transactions_dataset
from train_model import load_fraudulent, preprocess_fraudulent, ExpensesSummaryTransformer, load_forecast
import torch
from pytorch_forecasting.models.rnn import RecurrentNetwork
from pytorch_forecasting import TimeSeriesDataSet

def load_fraudulent_pipeline():
    pipeline_path = "models/fraudulent_pipeline.pkl"
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

def load_unlabeled_fraudulent_data():
    df = load_fraudulent()
    X, y = preprocess_fraudulent(df)
    X = X[y.isna()]
    return X

def predict_unlabeled_transactions():
    logging.info("Loading model and pipeline")
    pipeline = load_fraudulent_pipeline()

    logging.info("Loading unlabeled transactions")
    X = load_unlabeled_fraudulent_data()
    if X is None or X.empty:
        logging.info("No unlabeled transactions found.")
        return

    logging.info("Predicting labels")
    y_pred = pipeline.predict(X)

    # Create a dataframe with the predictions
    df = X.reset_index()["index"].copy()
    # convert to dataframe
    df = pd.DataFrame(df)
    # Add predictions to dataframe
    df["target"] = y_pred
    df["target"] = df[["target"]].map(lambda x: "Yes" if x == 1 else "No")
    df.set_index("index", inplace=True)

    # Convert to JSON format
    output_data = df.to_dict(orient='dict')
    # match expected format

    # Save the predictions
    output_path = "predictions/predictions_3.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    logging.info(f"Predictions saved to {output_path}")

def predict_next_three_months():
    # Load the trained forecast model
    kwargs, state_dict = torch.load('models/forecast_model.pth')
    model = RecurrentNetwork(**kwargs)
    model.load_state_dict(state_dict)
    model.eval()

    # Load client data
    with open("predictions_examples/predictions_4.json", "r") as f:
        predictions_example = json.load(f)

    client_ids = list(predictions_example["target"].keys())

    # convert to int
    client_ids = [int(client_id) for client_id in client_ids]
    # Temporarily drop all but two
    # client_ids = client_ids[:2]
    logging.info(f"Predicting for clients: {client_ids}")

    df = load_forecast()
    logging.info("df number of rows after filtering")
    logging.info(df["client_id"].isin(client_ids).sum())  

    
    df = df[df["client_id"].isin(client_ids)]

    # df.groupby("client_id").plot(x="date", y="outflows", title="Predicted outflows for next three months")
    # import matplotlib.pyplot as plt
    # plt.show()

    min_date = df["date"].min()
    df["date"] = ((df["date"] - min_date).dt.days/30).round().astype(int)


    # Append three months to the dataframe for each client
    df_dummy_1 = df.groupby("client_id").apply(lambda x: x.sort_values("date").iloc[-1]).reset_index(drop=True)
    df_dummy_1["date"] = df_dummy_1["date"] + 1

    # Create additional rows for the next two months
    df_dummy_2 = df_dummy_1.copy()
    df_dummy_2["date"] = df_dummy_2["date"] + 1
    df_dummy_3 = df_dummy_1.copy()
    df_dummy_3["date"] = df_dummy_3["date"] + 2

    # Concatenate the new rows with the original dataframe
    df = pd.concat([df, df_dummy_1, df_dummy_2, df_dummy_3]).fillna(0)
    # reset index
    df.reset_index(drop=True, inplace=True)

    # timeseries_dataset_params={
    #     "time_idx": "date",
    #     "target": "outflows",
    #     "group_ids": ["client_id"],
    #     "allow_missing_timesteps": True,
    #     "max_encoder_length": int(df["date"].max()),
    #     "min_encoder_length": 4,
    #     "max_prediction_length": 3,
    #     "time_varying_known_reals": ["date"],
    #     "time_varying_unknown_reals": ["inflows", "outflows", "net_cash_flow", "percentage_savings"],
    #     "predict_mode": True
    # }
    # dataset = TimeSeriesDataSet(df, **timeseries_dataset_params)           
    # model.dataset_parameters["predict_mode"] = True

    # Make predictions for the next three months
    predictions = model.predict(df, return_index=True, return_decoder_lengths=True)
    df_idx = pd.DataFrame(predictions.index)
    df_output = pd.DataFrame(predictions.output.cpu())
    df_prediction = pd.concat([df_idx, df_output], axis=1)
    df_prediction.columns = ["date", "client_id", "month_1", "month_2", "month_3"]

    logging.info("Predictions raw date")
    logging.info(df_prediction["date"])

    # Convert date from relative (int) to absolute (datetime)
    logging.info("Prediction dates")
    logging.info(df_prediction["date"])
    df_prediction["date"] = min_date + df_prediction["date"].apply(lambda x: pd.DateOffset(months=x))


    # transactions_df = get_transactions_dataset()
    # transactions_df = transactions_df[transactions_df["client_id"].isin(client_ids)]
    # transactions_df["date"] = pd.to_datetime(transactions_df["date"])
    logging.info("check if last dates are the same")
    logging.info(df.groupby("client_id")["date"].max())
    
    # Reshape the dataframe from wide to long format
    df_prediction = df_prediction.melt(id_vars=["date", "client_id"],
                                       value_vars=["month_1", "month_2", "month_3"],
                                       var_name="month", value_name="target")
    
    # Adjust the date column by adding the month offset from the melted dataframe
    df_prediction["date"] = df_prediction["date"] + \
                            (df_prediction["month"].str[-1].astype(int)).apply(lambda x: pd.DateOffset(months=x))

    df_prediction = df_prediction.drop(columns=["month"])
    df_prediction["date"] = pd.to_datetime(df_prediction["date"]).dt.strftime("%Y-%m")

    # Convert to JSON format
    output_data = df_prediction.groupby("client_id")
    output_data = output_data.apply(lambda x: x.set_index("date")["target"].to_dict()).to_dict()
    output_data = {"target": output_data}

    # Save the predictions
    logging.info("Saving predictions")
    output_path = "predictions/predictions_4.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    logging.info(f"Predictions saved to {output_path}")

    #change target to ourflows column
    df_prediction.columns = ["date", "client_id", "outflows"]
    # convert df date back to pandas datetime
    df["date"] = min_date + df["date"].apply(lambda x: pd.DateOffset(months=x))
    # drop last rows of df which are dummy
    dummy_length = 3*len(client_ids)
    df = df[:-dummy_length]
    # concatenate the original dataframe with the predictions
    df = pd.concat([df, df_prediction])
    # convert date to datetime
    df["date"] = pd.to_datetime(df["date"])
    logging.info("df after concatenation")
    logging.info(df.head())
    # df.groupby("client_id").plot(x="date", y="outflows", title="Predicted outflows for next three months")
    # # Save the plot
    # import matplotlib.pyplot as plt
    # plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # predict_unlabeled_transactions()

    predict_next_three_months()