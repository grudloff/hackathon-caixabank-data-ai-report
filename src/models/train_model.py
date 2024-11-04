from src.data.data_functions import get_fraud_labels, get_transactions_dataset, cash_flow_summary

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import logging

import pytorch_lightning as pl
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

def get_cash_flow_dataset(transactions_df: pd.DataFrame) -> pd.DataFrame:
    # get the cash flow for each client into a DataFrame
    client_ids = transactions_df["client_id"].unique()
    cash_flow_df = pd.DataFrame(columns=["date", "inflows", "outflows", "net_cash_flow", "percentage_savings"])
    cash_flow_df["client_id"] = client_ids
    cash_flow_df = cash_flow_df.set_index("client_id")
    for client_id in client_ids:
        cash_flow = cash_flow_summary(transactions_df, client_id)
        cash_flow_df.loc[client_id] = cash_flow
    return cash_flow_df

def load_forecast() -> pd.DataFrame:
    logging.info("Loading the data")
    transactions_df = get_transactions_dataset()
    df = get_cash_flow_dataset(transactions_df)
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

    # drop half of the data to speed up the training
    proportion = 0.3
    df_fraud = df[df["label"] == 1]
    df_non_fraud = df[df["label"] == 0].sample(frac=proportion)
    df = pd.concat([df_fraud, df_non_fraud])

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
    X = X.drop(columns=["date"])

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

    # drop missing labels
    X = X[~y.isna()]
    y = y[~y.isna()]
    # TODO: maybe assume missing labels are not fraudulent transactions

    return X, y

from sklearn.base import BaseEstimator, TransformerMixin

class ExpensesSummaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, start_date=None, end_date=None, num_expenses=1):
        self.start_date = start_date
        self.end_date = end_date
        self.num_expenses = num_expenses

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = get_transactions_dataset()
        client_ids = X["client_id"].unique()
        expenses_summary_df = pd.DataFrame(columns=self.get_feature_names_out(), dtype=float)
        # set index to client_id
        expenses_summary_df = expenses_summary_df.set_index("client_id")

        for client_id in client_ids:
            # check if client_id is in the transactions
            expenses = expenses_summary(df, client_id, self.start_date, self.end_date, plot=False,
                                        verbose=False)
            expenses = expenses.sort_values("Total Amount", ascending=False).head(self.num_expenses)
            expenses = expenses.to_numpy().flatten()
            # fill with nan if there are not enough expenses
            if len(expenses) < self.num_expenses * 6:
                expenses = np.concatenate([expenses, np.full(self.num_expenses * 6 - len(expenses), 0)])
            expenses_summary_df.loc[client_id] = expenses

        
        expenses_type_columns =[f"expenses_type_{i}" for i in range(self.num_expenses)]
        expenses_summary_df[expenses_type_columns] = expenses_summary_df[expenses_type_columns].astype("category")
        num_transactions_columns = [f"num_transactions_{i}" for i in range(self.num_expenses)]
        expenses_summary_df[num_transactions_columns] = expenses_summary_df[num_transactions_columns].astype(int)
        # set the rest to float
        float_columns = expenses_summary_df.columns.difference(expenses_type_columns + num_transactions_columns)
        expenses_summary_df[float_columns] = expenses_summary_df[float_columns].astype(float)

        # print types
        logging.info("Expenses summary types")
        logging.info(expenses_summary_df.dtypes)
        
        # use X["client_id"] to make expenses_summary_df have the same index
        expenses_summary_df = expenses_summary_df.reset_index()
        expenses_summary_df = X.join(expenses_summary_df.set_index("client_id"), on="client_id")

        return expenses_summary_df

    def get_feature_names_out(self, input_features=None):
        feature_names_out = ["client_id"]
        columns = self.generate_expense_columns()
        feature_names_out.extend(columns)
        return feature_names_out

    def generate_expense_columns(self):
        base_columns = ["Expenses Type", "Total Amount", "Average", "Max", "Min", "Num. Transactions"]
        base_columns = [col.lower().replace(".", "").replace(" ", "_") for col in base_columns]
        columns = [f"{col}_{i}" for i in range(self.num_expenses) for col in base_columns]
        return columns

class printer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info("Printing the data")
        logging.info(X)
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
    df = df.sample(frac=1) # shuffle the data
    X,y = preprocess_fraudulent(df)

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

    sample_weight = compute_sample_weight(class_weight="balanced", y=y)

    logging.info("Performing grid search")

    augmenter_pipeline = Pipeline([
        ("expenses_summary", ExpensesSummaryTransformer(start_date=None, end_date=None, num_expenses=3)),
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
            ("expenses_summary", augmenter_pipeline, ["client_id"])

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
    ).set_output(transform='pandas')

    pipeline = Pipeline([
        ("augmenter", augmenter),
        # ("printer", printer()),
        ("droper", droper),
        ("xgb", XGBClassifier(enable_categorical=True))
    ])

    param_grid = {
        "xgb__n_estimators": [200, 300],
        "xgb__max_depth": [5, 6],
        "xgb__learning_rate": [0.1, 0.01],
    }

    additional_params = {
        "xgb__sample_weight": sample_weight
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=1, verbose=3,
                               refit=False, scoring="f1")
    grid_search.fit(X, y, **additional_params)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    logging.info("Training the model")
    pipeline.set_params(**grid_search.best_params_)
    # pipeline.set_params(xgb__n_estimators=200, xgb__max_depth=5, xgb__learning_rate=0.1)
    pipeline.fit(X, y, **additional_params)

    y_pred = pipeline.predict(X)
    print("Classification report")
    print(classification_report(y, y_pred))

    # save the model
    logging.info("Saving the model")
    with open("models/fraudulent_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline

NUMBER_OF_MONTHS = 3

class ForecastModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=100, num_layers=1)
        self.linear = torch.nn.Linear(100, NUMBER_OF_MONTHS)
        self.loss = torch.nn.MSELoss()
    
    def forward(self, x):
        x, _ = self.lstm(x)
        # TODO: add client data 
        x = self.linear(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
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

    transactions = get_transactions_dataset()
    transactions.set_index("transaction_id", inplace=True)
    fraud_labels = get_fraud_labels()
    fraud_labels.set_index("transaction_id", inplace=True)
    df = transactions.join(fraud_labels, how="inner")
    # drop fraudulent transactions
    df = df[df["label"] == 0]
    df = df.drop(columns=["label"])

    last_month = df["date"].max()

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

    train_dataset = TimeSeriesDataSet(train_df, time_idx="date", target="amount", group_ids=["client_id"])
    val_dataset = TimeSeriesDataSet(val_df, time_idx="date", target="amount", group_ids=["client_id"])
    test_dataset = TimeSeriesDataSet(test_df, time_idx="date", target="amount", group_ids=["client_id"])

    train_dataloader = train_dataset.to_dataloader(train=True, batch_size=64)
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=64)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=64)

    model = ForecastModel()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloader, val_dataloader)

    trainer.test(test_dataloader)

    # save the model
    torch.save(model, "models/forecast_model.pth")

    return model

if __name__ == "__main__":

    # logging.basicConfig(level=logging.INFO)
    train_fraudulent() 

    
