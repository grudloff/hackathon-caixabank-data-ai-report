import pandas as pd
import json
from src.data.api_calls import get_client_data, get_cards_data
from functools import cache
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime

TRANSACTION_KEYS = ['transaction_id', 'date', 'client_id', 'card_id', 'amount', 'use_chip', 'merchant_id', 'merchant_city',
                    'merchant_state', 'zip', 'mcc', 'errors']
CLIENT_KEYS = ['client_id', 'address', 'birth_month', 'birth_year', 'credit_score', 'current_age', 'gender', 'latitude',
               'longitude', 'num_credit_cards', 'per_capita_income', 'retirement_age', 'total_debt', 'yearly_income']
CARD_KEYS = ['card_id', 'acct_open_date', 'card_brand', 'card_number', 'card_on_dark_web', 'card_type', 'client_id',
             'credit_limit', 'cvv', 'expires', 'has_chip', 'num_cards_issued', 'year_pin_last_changed']

def get_transactions_dataset() -> pd.DataFrame:
    """
    Get the transactions dataset from the data/raw/transactions_data.csv.

    Returns
    -------
    pandas DataFrame
        DataFrame with the transactions data.
    """
    logging.info("Reading transactions data from file")
    filepath = "data/raw/transactions_data.csv"
    transactions_df = pd.read_csv(filepath, names=TRANSACTION_KEYS, index_col=None, header=0)
    
    logging.info("Cleaning transactions data")
    transactions_df["date"] = pd.to_datetime(transactions_df["date"])
    transactions_df["amount"] = transactions_df["amount"].str.replace("$", "").astype(float)
    transactions_df["amount"] = pd.to_numeric(transactions_df["amount"], errors="coerce")
    transactions_df["zip"] = pd.to_numeric(transactions_df["zip"], errors="coerce")
    transactions_df["mcc"] = pd.to_numeric(transactions_df["mcc"], errors="coerce")
    transactions_df["client_id"] = pd.to_numeric(transactions_df["client_id"], errors="coerce")
    transactions_df["card_id"] = pd.to_numeric(transactions_df["card_id"], errors="coerce")
    transactions_df["merchant_id"] = pd.to_numeric(transactions_df["merchant_id"], errors="coerce")

    return transactions_df

def get_clients_dataset() -> pd.DataFrame:
    """
    Get the clients dataset from the data/raw/clients_data.csv.

    Returns
    -------
    pandas DataFrame
        DataFrame with the clients data.
    """
    try:
        logging.info("Reading clients data from file")
        clients_df = pd.read_csv("data/raw/clients_data.csv")
    except FileNotFoundError:
        logging.info("File not found")
        logging.info("Fetching data from the API")
        client_ids = get_transactions_dataset()["client_id"].unique()
        clients_data = []
        for client_id in tqdm(client_ids):
            client_data = get_client_data(client_id)[client_id]
            client_data["client_id"] = client_id
            clients_data.append(client_data)
        clients_df = pd.DataFrame(clients_data, columns=CLIENT_KEYS)

        logging.info("Saving clients data to file")
        clients_df.to_csv("data/raw/clients_data.csv")
        pass

    logging.info("Cleaning clients data")
    clients_df["birth_month"] = pd.to_numeric(clients_df["birth_month"], errors="coerce")
    clients_df["birth_year"] = pd.to_numeric(clients_df["birth_year"], errors="coerce")
    clients_df["credit_score"] = pd.to_numeric(clients_df["credit_score"], errors="coerce")
    clients_df["current_age"] = pd.to_numeric(clients_df["current_age"], errors="coerce")
    clients_df["latitude"] = pd.to_numeric(clients_df["latitude"], errors="coerce")
    clients_df["longitude"] = pd.to_numeric(clients_df["longitude"], errors="coerce")
    clients_df["num_credit_cards"] = pd.to_numeric(clients_df["num_credit_cards"], errors="coerce")
    clients_df["per_capita_income"] = clients_df["per_capita_income"].str.replace("$", "")
    clients_df["per_capita_income"] = pd.to_numeric(clients_df["per_capita_income"], errors="coerce")
    clients_df["retirement_age"] = pd.to_numeric(clients_df["retirement_age"], errors="coerce")
    clients_df["total_debt"] = clients_df["total_debt"].str.replace("$", "").astype(float)
    clients_df["total_debt"] = pd.to_numeric(clients_df["total_debt"], errors="coerce")
    clients_df["yearly_income"] = clients_df["yearly_income"].str.replace("$", "").astype(float)
    clients_df["yearly_income"] = pd.to_numeric(clients_df["yearly_income"], errors="coerce")
    
    return clients_df

def get_cards_dataset() -> pd.DataFrame:
    """
    Get the cards dataset from the data/raw/cards_data.csv.

    Returns
    -------
    pandas DataFrame
        DataFrame with the cards data.
    """
    try:
        logging.info("Reading cards data from file")
        cards_df = pd.read_csv("data/raw/cards_data.csv")
    except FileNotFoundError:
        logging.info("File not found")
        logging.info("Fetching data from the API")
        client_ids = get_transactions_dataset()["client_id"].unique()
        cards_data = []
        for client_id in tqdm(client_ids):
            card_data = get_cards_data(client_id)
            for card_id, data in card_data.items():
                data["card_id"] = card_id
                cards_data.append(data)
        cards_df = pd.DataFrame(cards_data, columns=CARD_KEYS)
        logging.info("Saving cards data to file")
        cards_df.to_csv("data/raw/cards_data.csv")
        pass    

    logging.info("Cleaning cards data")
    cards_df["acct_open_date"] = pd.to_datetime(cards_df["acct_open_date"], format="%m/%Y")
    cards_df["credit_limit"] = cards_df["credit_limit"].str.replace("$", "").astype(float)
    cards_df["credit_limit"] = pd.to_numeric(cards_df["credit_limit"], errors="coerce")
    cards_df["expires"] = pd.to_datetime(cards_df["expires"], format="%m/%Y")
    cards_df["num_cards_issued"] = pd.to_numeric(cards_df["num_cards_issued"], errors="coerce")
    cards_df["client_id"] = pd.to_numeric(cards_df["client_id"], errors="coerce")

    return cards_df


def get_mcc_codes() -> dict:
    """
    Get the MCC codes from the data/raw/mcc_codes.json.

    Returns
    -------
    dict
        Dictionary with the MCC codes.
    """
    filepath = "data/raw/mcc_codes.json"
    with open(filepath, "r") as file:
        return json.load(file)

def get_fraud_labels() -> pd.DataFrame:
    """
    Get the fraud labels from the data/raw/fraud_labels.json.

    Returns
    -------
    list
        List with the fraud labels.
    """
    filepath = "data/raw/train_fraud_labels.json"
    result = pd.read_json(filepath)
    result = result.rename(columns={"target": "label"})
    result = result.reset_index().rename(columns={"index": "transaction_id"})
    result["transaction_id"] = result["transaction_id"].astype(int)
    result["label"] = result["label"].map({"No": 0, "Yes": 1}).astype(int)
    return result

def earnings_and_expenses(df: pd.DataFrame, client_id: int, start_date: str, end_date: str, plot=True, verbose=True) -> pd.DataFrame:
    """
    For the period defined in between start_date and end_date (both included), get the client data available and return
    a pandas DataFrame with the Earnings and Expenses total amount for the period range and user given.The expected columns are:
        - Earnings
        - Expenses
    The DataFrame should have the columns in this order ['Earnings','Expenses']. Round the amounts to 2 decimals.

    Create a Bar Plot with the Earnings and Expenses absolute values and save it as "reports/figures/earnings_and_expenses.png" .

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame of the data to be used for the agent.
    client_id : int
        Id of the client.
    start_date : str
        Start date for the date period. In the format "YYYY-MM-DD".
    end_date : str
        End date for the date period. In the format "YYYY-MM-DD".


    Returns
    -------
    Pandas Dataframe with the earnings and expenses rounded to 2 decimals.

    """

    if not verbose:
        # save current logging level
        level = logging.getLogger().getEffectiveLevel()
        # set logging level to ERROR
        logging.getLogger().setLevel(logging.ERROR)

    logging.info("Starting earnings and expenses")
    logging.info("Preprocessing data")
    if isinstance(df["amount"].iloc[0], str):
        df["amount"] = df["amount"].str.replace("$", "").astype(float)

    logging.info("Filtering data")
    client_df = df[df["client_id"] == client_id]
    client_df = client_df[(client_df["date"] >= start_date) & (client_df["date"] <= end_date)]
    client_df = client_df[["amount"]]
    logging.info("Calculating earnings and expenses")
    earnings = client_df[client_df["amount"] > 0]["amount"].sum()
    expenses = client_df[client_df["amount"] < 0]["amount"].sum()
    earnings = round(earnings, 2)
    expenses = round(expenses, 2)

    if plot:
        logging.info("Creating bar plot")
        earnings_df = pd.DataFrame({"Earnings": [earnings], "Expenses": [expenses]})
        earnings_df.plot.bar(title="Earnings and Expenses")
        plt.ylabel("Amount")
        plt.xticks(rotation=0)
        plt.legend().set_visible(False)
        logging.info("Saving plot")
        if not os.path.exists("reports/figures"):
            os.makedirs("reports/figures")
        plt.savefig("reports/figures/earnings_and_expenses.png")
    
    if not verbose:
        # set logging level back to previous level
        logging.getLogger().setLevel(level)

    return pd.DataFrame({"Earnings": [earnings], "Expenses": [expenses]})


def expenses_summary(df: pd.DataFrame, client_id: int, start_date: str=None, end_date: str=None,
                     plot=True, verbose=True) -> pd.DataFrame:
    """
    For the period defined in between start_date and end_date (both included), get the client data available and return
    a Pandas Data Frame with the Expenses by merchant category. The expected columns are:
        - Expenses Type --> (merchant category names)
        - Total Amount
        - Average
        - Max
        - Min
        - Num. Transactions
    The DataFrame should be sorted alphabeticaly by Expenses Type and values have to be rounded to 2 decimals. Return the dataframe with the columns in the given order.
    The merchant category names can be found in data/raw/mcc_codes.json .

    Create a Bar Plot with the data in absolute values and save it as "reports/figures/expenses_summary.png" .

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame  of the data to be used for the agent.
    client_id : int
        Id of the client.
    start_date : str
        Start date for the date period. In the format "YYYY-MM-DD".
    end_date : str
        End date for the date period. In the format "YYYY-MM-DD".


    Returns
    -------
    Pandas Dataframe with the Expenses by merchant category.

    """
    if not verbose:
        # save current logging level
        level = logging.getLogger().getEffectiveLevel()
        # set logging level to ERROR
        logging.getLogger().setLevel(logging.ERROR)
    logging.info("Starting expenses summary")
    logging.info("Preprocesing data")
    if isinstance(df["amount"].iloc[0], str):
        df["amount"] = df["amount"].str.replace("$", "").astype(float)

    logging.info("Filtering data")
    client_df = df[df["client_id"] == client_id]
    if start_date is not None and end_date is not None:
        client_df = client_df[(client_df["date"] >= start_date) & (client_df["date"] <= end_date)]
    elif start_date is not None:
        client_df = client_df[client_df["date"] >= start_date]
    elif end_date is not None:
        client_df = client_df[client_df["date"] <= end_date]
    client_df = client_df[client_df["amount"] < 0]
    client_df["amount"] = client_df["amount"].abs()
    client_df = client_df[["mcc", "amount"]]

    logging.info("Mapping merchant category names")
    client_df["mcc"] = client_df["mcc"].astype(str)
    client_df["mcc"] = client_df["mcc"].map(get_mcc_codes())

    logging.info("Grouping by merchant category")
    logging.info("Calculating total amount, average, max, min and number of transactions")
    grouped_df = client_df.groupby("mcc")["amount"].agg(["sum", "mean", "min", "max", "count"])
    grouped_df = grouped_df.round(2)
    grouped_df = grouped_df.sort_index()
    grouped_df = grouped_df.reset_index()
    grouped_df.columns = ["Expenses Type", "Total Amount", "Average", "Max", "Min", "Num. Transactions"]
    logging.info("Expenses summary finished")

    if plot:
        logging.info("Creating bar plot")
        grouped_df.plot.bar(x="Expenses Type", y="Total Amount", title="Expenses Summary")
        plt.ylabel("Total Amount")
        plt.xticks(rotation=90)
        plt.legend().set_visible(False)
        logging.info("Saving plot")
        if not os.path.exists("reports/figures"):
            os.makedirs("reports/figures")
        plt.savefig("reports/figures/expenses_summary.png")
        logging.info("Plot saved")
    
    if not verbose:
        # set logging level back to previous level
        logging.getLogger().setLevel(level)

    return grouped_df


def cash_flow_summary(
    df: pd.DataFrame, client_id: int, start_date: str=None, end_date: str=None, verbose=True
) -> pd.DataFrame:
    """
    For the period defined by start_date and end_date (both inclusive), retrieve the available client data and return a Pandas DataFrame containing cash flow information.

    If the period exceeds 60 days, group the data by month, using the end of each month for the date. If the period is 60 days or shorter, group the data by week.

        The expected columns are:
            - Date --> the date for the period. YYYY-MM if period larger than 60 days, YYYY-MM-DD otherwise.
            - Inflows --> the sum of the earnings (positive amounts)
            - Outflows --> the sum of the expenses (absolute values of the negative amounts)
            - Net Cash Flow --> Inflows - Outflows
            - % Savings --> Percentage of Net Cash Flow / Inflows

        The DataFrame should be sorted by ascending date and values rounded to 2 decimals. The columns should be in the given order.

        Parameters
        ----------
        df : pandas DataFrame
           DataFrame  of the data to be used for the agent.
        client_id : int
            Id of the client.
        start_date : str
            Start date for the date period. In the format "YYYY-MM-DD".
        end_date : str
            End date for the date period. In the format "YYYY-MM-DD".


        Returns
        -------
        Pandas Dataframe with the cash flow summary.

    """
    MONTHLY_THRESHOLD = 60

    if not verbose:
        # save current logging level
        level = logging.getLogger().getEffectiveLevel()
        # set logging level to ERROR
        logging.getLogger().setLevel(logging.ERROR)

    logging.info("Starting cash flow summary")
    logging.info("Preprocessing data")
    if isinstance(df["amount"].iloc[0], str):
        df["amount"] = df["amount"].str.replace("$", "").astype(float)

    logging.info("Filtering data")
    client_df = df[df["client_id"] == client_id]
    if start_date is None:
        start_date = pd.to_datetime(client_df["date"].min()).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = pd.to_datetime(client_df["date"].max()).strftime("%Y-%m-%d")
    client_df = client_df[(client_df["date"] >= start_date) & (client_df["date"] <= end_date)]
    client_df = client_df[["date", "amount"]]
    client_df["date"] = pd.to_datetime(client_df["date"])

    client_df["earnings"] = client_df["amount"].apply(lambda x: x if x > 0 else 0)
    client_df["expenses"] = client_df["amount"].apply(lambda x: -x if x < 0 else 0)
    client_df = client_df.drop("amount", axis=1)

    logging.info("Calculating cash flow summary")
    logging.info("Aggregating by date")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    monthly = (end_datetime - start_datetime).days > MONTHLY_THRESHOLD
    if monthly:
        logging.info("Grouping by month")
        period = "M"
    else:
        logging.info("Grouping by week")
        period = "W"
    client_df["date"] = client_df["date"].dt.to_period(period)
    client_df = client_df.groupby("date").agg(["sum"])

    logging.info("Resetting index")
    client_df = client_df.reset_index()
    client_df.columns = client_df.columns.to_flat_index()
    client_df.columns = client_df.columns.map(lambda x: x[0])

    logging.info("Formatting date")
    if monthly:
        # timedelta to month in %YYYY-%MM format
        client_df["date"] = client_df["date"].dt.to_timestamp(how="end")
        client_df["date"] = client_df["date"].dt.strftime("%Y-%m")
    else:
        # timedelta to week in %YYYY-%MM-%DD format
        client_df["date"] = client_df["date"].dt.to_timestamp(how="end")
        client_df["date"] = client_df["date"].dt.strftime("%Y-%m-%d")
    
    logging.info("Calculating net cash flow and % savings")
    client_df["net_cash_flow"] = client_df["earnings"] - client_df["expenses"]
    client_df["% savings"] = client_df["net_cash_flow"] / client_df["earnings"] * 100
    logging.info("Rounding values")
    client_df = client_df.round(2)
    client_df.columns = ["Date", "Inflows", "Outflows", "Net Cash Flow", "% Savings"]
    logging.info("Cash flow summary finished")

    if not verbose:
        # set logging level back to previous level
        logging.getLogger().setLevel(level)

    return client_df


if __name__ == "__main__":
    transactions_df = get_transactions_dataset()

    logging.getLogger().setLevel(logging.INFO)

    client_id = 126
    start_date = "2013-01-01"
    end_date = "2013-03-28"

    logging.info("-------------------------------------------------")
    expenses_df = expenses_summary(transactions_df, client_id, start_date, end_date)
    logging.info("Expenses Summary")
    logging.info(expenses_df)

    logging.info("-------------------------------------------------")
    cash_flow_df = cash_flow_summary(transactions_df, client_id, start_date, end_date)
    logging.info("Cash Flow Summary")
    logging.info(cash_flow_df)

    logging.info("-------------------------------------------------")
    earnings_df = earnings_and_expenses(transactions_df, client_id, start_date, end_date)
    logging.info("Earnings and Expenses")
    logging.info(earnings_df)

