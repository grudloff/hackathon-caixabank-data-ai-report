from src.data.data_functions import (get_fraud_labels, get_transactions_dataset, cash_flow_summary, earnings_and_expenses,
                                     get_clients_dataset)

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import logging

import lightning.pytorch as pl
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet

from src.data.data_functions import expenses_summary

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

import numpy as np

import os

import pickle

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from functools import partial

from multiprocessing import Pool

from pytorch_forecasting.models.rnn import RecurrentNetwork

from lightning.pytorch.tuner import Tuner

def get_cash_flow_dataset(transactions_df: pd.DataFrame) -> pd.DataFrame:
    # get the cash flow for each client into a DataFrame
    logging.info("Calculating cash flow for each client")
    client_ids = transactions_df["client_id"].unique()
    cash_flow_col_names = ["date", "inflows", "outflows", "net_cash_flow", "percentage_savings", "client_id"]
    cash_flow_df = pd.DataFrame(columns=cash_flow_col_names)
    for client_id in client_ids:
        cash_flow = cash_flow_summary(transactions_df, client_id, verbose=False)
        cash_flow["client_id"] = client_id
        cash_flow["client_id"] = cash_flow["client_id"].astype(int)
        cash_flow.columns = cash_flow_col_names
        cash_flow["date"] = pd.to_datetime(cash_flow["date"])
        cash_flow_df = pd.concat([cash_flow_df, cash_flow])
    cash_flow_df = cash_flow_df.reset_index(drop=True)
    return cash_flow_df

def load_forecast(use_client_info: bool = False, add_month: bool=False) -> pd.DataFrame:
    logging.info("Loading the data")
    transactions = get_transactions_dataset()
    transactions.set_index("transaction_id", inplace=True)
    fraud_labels = get_fraud_labels()
    fraud_labels.set_index("transaction_id", inplace=True)
    df = transactions.join(fraud_labels, how="inner")
    logging.info("Dropping fraudulent transactions")
    # drop fraudulent transactions or unknown labels
    # TODO: could use result from fraudulent model
    df = df[df["label"] == 0]
    df = df[~df["label"].isna()]
    df = df.drop(columns=["label"])
    df = get_cash_flow_dataset(df)
    # augment data
    if add_month:
        logging.info("Adding month information")
        df["month"] = df["date"].dt.month
    if use_client_info:
        logging.info("Adding client information")
        client_info = get_clients_dataset()
        df = df.merge(client_info, on="client_id", how="left")
    return df

def load_fraudulent() -> pd.DataFrame:
    logging.info("Loading the data")

    try:
        df = pd.read_csv("data/processed/transactions_with_labels.csv")
    except FileNotFoundError:
        transactions = get_transactions_dataset()
        fraud_labels = get_fraud_labels()
        df = transactions.merge(fraud_labels, on='transaction_id', how='left')
        if not os.path.exists("data/processed"):
            os.makedirs("data/processed")
        df.to_csv("data/processed/transactions_with_labels.csv")
    
    return df

def preprocess_fraudulent(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    X = df.drop(columns=["label"])
    y = df["label"]

    # print dtypes
    logging.info("Data types")
    logging.info("X dtypes")
    logging.info(X.dtypes)
    logging.info("y dtypes")
    logging.info(y.dtypes)

    # preprocess date
    logging.info("Preprocessing date")
    X["date"] = pd.to_datetime(X["date"])
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    # X = X.drop(columns=["date"])

    # scale numerical columns
    logging.info("Scaling numerical columns")
    numerical_columns = X.select_dtypes(include=["float64", "int64"]).columns
    # remove client_id
    numerical_columns = numerical_columns.drop(["client_id", "transaction_id"])
    scaler = MinMaxScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # preprocess categorical columns
    logging.info("Preprocessing categorical columns")

    # check how many transactions have errors
    logging.info("Number of transactions with errors")
    logging.info(X["errors"].value_counts())

    X["errors"] = X["errors"].fillna("")
    X["errors"] = X["errors"].map(lambda x: len(x)) 

    categorical_columns = X.select_dtypes(include=["object"]).columns
    # X = pd.get_dummies(X, columns=categorical_columns)
    # convert to categorical type
    X[categorical_columns] = X[categorical_columns].astype("category")

    # drop errors column
    # X = X.drop(columns=["errors"])

    return X, y

from sklearn.base import BaseEstimator, TransformerMixin

class ExpensesSummaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_expenses=1, time_window=3, disable_progress_bar=False, multiprocessing=False):
        self.num_expenses = num_expenses
        self.time_window = time_window
        self.disable_progress_bar = disable_progress_bar
        self.multiprocessing = multiprocessing

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert "client_id" in X.columns
        assert "date" in X.columns
        assert len(X.columns) == 2
        df = get_transactions_dataset()
        expenses_summary_df = pd.DataFrame(columns=self.get_feature_names_out()[1:], dtype=float)

        # TODO: should be adding client data also!
        if self.multiprocessing:
            # NOTE: multiprocessing works slower, maybe with a large number of cores the tradeoff is worth it
            with Pool() as pool:
                func = partial(self.add_expenses_summary_row, df, expenses_summary_df,
                               time_window=self.time_window, num_expenses=self.num_expenses)
                # NOTE: imap requires a pickable iterator, so we need to convert the DataFrame to a numpy array
                X = X.reset_index()
                X_np = X.to_numpy()
                for _ in tqdm(pool.imap_unordered(func, X_np), total=len(X),
                              disable=self.disable_progress_bar):
                    pass
                # set index
                X = X.set_index("Index")
        else:
            for row in tqdm(X.itertuples(), disable=self.disable_progress_bar):
                self.add_expenses_summary_row(df, expenses_summary_df, row, self.time_window, self.num_expenses)

        # set dtypes
        expenses_type_columns =[f"expenses_type_{i}" for i in range(self.num_expenses)]
        expenses_summary_df[expenses_type_columns] = expenses_summary_df[expenses_type_columns].astype("category")
        num_transactions_columns = [f"num_transactions_{i}" for i in range(self.num_expenses)]
        expenses_summary_df[num_transactions_columns] = expenses_summary_df[num_transactions_columns].astype(int)
        float_columns = expenses_summary_df.columns.difference(expenses_type_columns + num_transactions_columns)
        expenses_summary_df[float_columns] = expenses_summary_df[float_columns].astype(float)

        assert expenses_summary_df.isna().sum().sum() == 0

        expenses_summary_df = X[["client_id"]].join(expenses_summary_df)
        return expenses_summary_df

    @staticmethod
    def add_expenses_summary_row(df, expenses_summary_df, row, time_window, num_expenses):
        if isinstance(row, tuple) or isinstance(row, np.ndarray):
            idx, client_id, date = row
        else:
            idx, client_id, date = getattr(row, "Index"), getattr(row, "client_id"), getattr(row, "date")
        start_date = date - pd.DateOffset(months=time_window)
        end_date = date
        earnings_expenses = earnings_and_expenses(df, client_id, start_date, end_date, plot=False,
                                                      verbose=False)
        earnings_expenses = earnings_expenses.to_numpy().flatten()
        assert len(earnings_expenses) == 2
        expenses_by_mcc = expenses_summary(df, client_id, start_date, end_date, plot=False,
                                        verbose=False)
        expenses_by_mcc = expenses_by_mcc.sort_values("Total Amount", ascending=False).head(num_expenses)
        expenses_by_mcc = expenses_by_mcc.to_numpy().flatten()
        if len(expenses_by_mcc) < num_expenses * 6:
            expenses_by_mcc = np.concatenate([expenses_by_mcc, np.full(num_expenses * 6 - len(expenses_by_mcc), 0)])
        expenses_summary_df.loc[idx] = np.concatenate([earnings_expenses, expenses_by_mcc])

    def get_feature_names_out(self, input_features=None):
        client_column = ["client_id"]
        earnings_expenses_cloumns = ["Earnings", "Expenses"]
        expense_mcc_columns = self.generate_expense_mcc_columns()
        return client_column + earnings_expenses_cloumns + expense_mcc_columns

    def generate_expense_mcc_columns(self):
        base_columns = ["Expenses Type", "Total Amount", "Average", "Max", "Min", "Num. Transactions"]
        base_columns = [col.lower().replace(".", "").replace(" ", "_") for col in base_columns]
        columns = [f"{col}_{i}" for i in range(self.num_expenses) for col in base_columns]
        return columns

class printer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("Printing the data")
        print(X.head())
        return X

class dummy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

def train_fraudulent():

    logging.info("Training the fraudulent transaction model")

    df = load_fraudulent()
    # print value counts for label
    logging.info("Value counts for label")
    logging.info(df["label"].value_counts())

    proportion = 0.01
    df_fraud = df[df["label"] == 1]
    # NOTE: we are undersampling the non-fraudulent transactions
    # to make the training faster while not losing too much information
    df_non_fraud = df[df["label"] == 0].sample(frac=proportion)
    df = pd.concat([df_fraud, df_non_fraud])
    df = df.sample(frac=1) # shuffle the data

    X,y = preprocess_fraudulent(df)

    # drop missing labels
    X = X[~y.isna()]
    y = y[~y.isna()]

    # print dtypes
    logging.info("Data types")
    logging.info("X dtypes")    
    logging.info(X.dtypes)
    logging.info("y dtypes")
    logging.info(y.dtypes)

    logging.info("Calculating class weights")
    # class_weights = y.value_counts(normalize=True)
    # class_weights = class_weights[0] / class_weights
    # class_weights = class_weights.to_dict()
    # class_weights = list(class_weights.values())

    # sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    scale_pos_weight = y.value_counts()[1] / y.value_counts()[0]

    logging.info("Performing grid search")

    augmenter_pipeline = Pipeline([
        ("expenses_summary", ExpensesSummaryTransformer(num_expenses=3, time_window=3)),
        ("preprocessor", ColumnTransformer([
            ("numerical", MinMaxScaler(), make_column_selector(dtype_include=["float64", "int64"]))
            ],
            remainder="passthrough",
            verbose_feature_names_out=False)
            )
    ])
    
    augmenter_pipeline.set_output(transform='pandas')

    augmenter = ColumnTransformer(
        transformers = [
            ("expenses_summary", augmenter_pipeline, ["client_id", "date"]),

        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    augmenter.set_output(transform='pandas')

    droper = ColumnTransformer(
    transformers=[
        ('column_dropper', 'drop', ["client_id", "transaction_id"]),
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
    )
    
    droper.set_output(transform='pandas')

    partial_pipeline = Pipeline([
        ("augmenter", augmenter),
        # ("printer", printer()),
        ("droper", droper),
        # ("xgb", XGBClassifier(enable_categorical=True, scale_pos_weight=scale_pos_weight))
        # ("xgb", XGBClassifier(enable_categorical=True))

    ])

    if os.path.exists("data/processed/fraudulent_dataset.pkl") and os.path.exists("models/fraudulent_partial_pipeline.pkl"):
        logging.info("Loading the dataset")
        X = pd.read_pickle("data/processed/fraudulent_dataset.pkl")
        with open("models/fraudulent_partial_pipeline.pkl", "rb") as f:
            partial_pipeline = pickle.load(f)
    else:
        logging.info("Transforming the dataset")
        X = partial_pipeline.fit_transform(X)
        X.to_pickle("data/processed/fraudulent_dataset.pkl")
        with open("models/fraudulent_partial_pipeline.pkl", "wb") as f:
            pickle.dump(partial_pipeline, f)

    model = XGBClassifier(enable_categorical=True, scale_pos_weight=scale_pos_weight)

    param_grid = {
        "n_estimators": [400],
        "max_depth": [7],
        "learning_rate": [0.5],
        "gamma": [0.1, 0.3],
        "min_child_weight" : [1, 3],
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=1, verbose=3,
                               refit=False, scoring="f1")
    grid_search.fit(X, y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    logging.info("Training the model")
    # traintest split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # pipeline.set_params(**grid_search.best_params_)
    model.set_params(n_estimators=200, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification report")
    print(classification_report(y_test, y_pred))

    # train on full dataset
    model.fit(X, y, verbose=3)

    # convert model to pipeline including the partial pipeline
    partial_pipeline.set_params(augmenter__expenses_summary__expenses_summary__disable_progress_bar=True)
    pipeline = Pipeline([
        ("partial_pipeline", partial_pipeline),
        ("model", model)
    ])

    # save the model
    logging.info("Saving the model")
    with open("models/fraudulent_model.pkl", "wb") as f:
        pickle.dump(model, f)
    # save the pipeline
    with open("models/fraudulent_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    return model

NUMBER_OF_MONTHS = 3

# class ForecastModel(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.lstm = torch.nn.LSTM(input_size=1, hidden_size=100, num_layers=1)
#         self.linear = torch.nn.Linear(100, NUMBER_OF_MONTHS)
#         self.loss = torch.nn.MSELoss()
    
#     def forward(self, x):
#         x, _ = self.lstm(x)
#         # TODO: add client data 
#         x = self.linear(x)
#         return x
    
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
#         return loss
    
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=0.001)
    
ALL_COLUMNS = ['transaction_id', 'date', 'client_id', 'card_id', 'amount', 'use_chip', 'merchant_id', 'merchant_city',
                    'merchant_state', 'zip', 'mcc', 'errors']
NUMERICAL_COLUMNS = ['amount']
CATEGORICAL_COLUMNS = ['use_chip', 'merchant_id', 'merchant_city', 'merchant_state', 'zip', 'mcc']
DATE_COLUMNS = ['date']

# class ClientForecastDataset(torch.utils.data.Dataset):
#     def __init__(self, transactions):
#         self.scaler = MinMaxScaler()
#         self.transactions = self.scaler.fit_transform(transactions[NUMERICAL_COLUMNS])
#         self.transactions = pd.get_dummies(transactions, columns=CATEGORICAL_COLUMNS)
#         self.transactions["date"] = self.transactions["date"].dt.month
#         self.client_ids = transactions["client_id"].unique()
#         self.transactions = self.transactions.drop(columns=["errors"])
    
#     def __len__(self):
#         return len(self.client_ids)
    
#     def __getitem__(self, idx):
#         client_id = self.client_ids[idx]
#         client_transactions = self.transactions[self.transactions["client_id"] == client_id]
#         client_transactions = client_transactions.sort_values("date")
#         X = client_transactions.drop(columns=["client_id"])
#         y = X["amount"].iloc[-NUMBER_OF_MONTHS:]
#         X = X.iloc[:-NUMBER_OF_MONTHS]
#         return torch.tensor(X.values), torch.tensor(y.values)
    
#     # TODO: check if this works (or something like this)
#     def collate_fn(self, batch):
#         X, y = zip(*batch)
#         X = torch.stack(X)
#         y = torch.stack(y)
#         return X, y

def train_forecast():
    """
    Train a model to forecast the amount of money that will be spent in the next three months
    by a client, given that its historic transactions are known.

    For this an LSTM model will be used.
    """

    df = load_forecast(use_client_info=True, add_month=True)

    # print columns 
    logging.info("Columns")
    logging.info(df.columns)
    # dtypes
    logging.info("Data types")
    logging.info(df.dtypes)

    # print head
    logging.info("Head")
    logging.info(df.head())

    last_month = df["date"].max()
    last_month = pd.to_datetime(last_month)

    clients = df["client_id"].unique()
    n_clients = len(clients)
    n_val = n_clients // 10
    n_test = n_clients // 10
    n_train = n_clients - n_val - n_test

    train_clients = clients[:n_train]
    val_clients = clients[n_train:n_train+n_val]
    test_clients = clients[n_train+n_val:]

    train_df = df[df["client_id"].isin(train_clients)]
    val_df = df[df["client_id"].isin(val_clients)]
    test_df = df[df["client_id"].isin(test_clients)]

    train_df = train_df[train_df["date"] < last_month - pd.DateOffset(months=9)]
    val_df = val_df[val_df["date"] < last_month - pd.DateOffset(months=6)]
    test_df = test_df[test_df["date"] < last_month - pd.DateOffset(months=3)]

    # convert date to integer representation with zero being the first date
    print(train_df.date.value_counts())
    min_date = df["date"].min()
    for dataset in [train_df, val_df, test_df]:
        dataset["date"] = ((dataset["date"] - min_date).dt.days/30).round().astype(int)
    print(train_df.date.value_counts())
    print(train_df["date"].max())

    # Define static features
    static_reals = [
        'birth_month', 'birth_year', 'credit_score', 'current_age',
        'latitude', 'longitude', 'num_credit_cards',
        'per_capita_income', 'retirement_age', 'total_debt',
        'yearly_income'
    ]

    timeseries_dataset_params={
        "time_idx": "date",
        "target": "outflows",
        "group_ids": ["client_id"],
        "static_reals": static_reals,
        "allow_missing_timesteps": True,
        "max_encoder_length": int(train_df["date"].max()),
        "min_encoder_length": 3,
        "max_prediction_length": 3,
        "time_varying_known_reals": ["date", "month"],
        "time_varying_unknown_reals": ["inflows", "outflows", "net_cash_flow", "percentage_savings"],
    }
    train_dataset = TimeSeriesDataSet(train_df, **timeseries_dataset_params)
    val_dataset = TimeSeriesDataSet(val_df, **timeseries_dataset_params)
    test_dataset = TimeSeriesDataSet(test_df, **timeseries_dataset_params)

    train_dataloader = train_dataset.to_dataloader(train=True, batch_size=128, num_workers=9)
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=128, num_workers=9)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=128, num_workers=9)

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)],
    )

    model = RecurrentNetwork.from_dataset(
        train_dataset,
        cell_type="LSTM",
        hidden_size=100,
        time_varying_reals_decoder=["date", "inflows", "net_cash_flow", "percentage_savings"],
    )

    # find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
    res = Tuner(trainer).lr_find(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, max_lr=1,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()

    logging.info("Training the model")
    trainer.fit(model, train_dataloader, val_dataloader)

    logging.info("Saving the model")
    # trainer.save_checkpoint("models/forecast_model.ckpt")
    torch.save([model._hparams, model.state_dict()], 'models/forecast_model.pth')

    logging.info("Testing the model")
    trainer.test(model, test_dataloader)

    return model

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    # train_fraudulent()

    train_forecast() 

    
