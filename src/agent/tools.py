import pandas as pd

from langchain_core.tools import tool
from src.data.data_functions import earnings_and_expenses, expenses_summary, cash_flow_summary

import logging

# convert data_functions to tools
earnings_and_expenses = tool(earnings_and_expenses)
expenses_summary = tool(expenses_summary)
cash_flow_summary = tool(cash_flow_summary)

@tool
def get_days_in_month(month: int, year: int) -> int:
    """
    Get the number of days in a month.

    Parameters
    ----------
    month : int
        The month as an integer.
    year : int
        The year.

    Returns
    -------
    int
        Number of days in the month.
    """
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    elif month == 2:
        if year % 4 == 0:
            return 29
        else:
            return 28
    else:
        return 0

@tool
def check_client_data_exists(df: pd.DataFrame, client_id) -> bool:
    """
    Check if data exists for a given client_id.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with the data.
    client_id : int
        The client_id to check.

    Returns
    -------
    bool
        True if data exists, False otherwise.
    """
    return client_id in df["client_id"].unique()

@tool
def check_dates_exist_for_client(df: pd.DataFrame, client_id: int, start_date: str, end_date: str) -> bool:
    """
    Check if data exists for a given client_id and date range.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with the data.
    client_id : int
        The client_id to check.
    start_date : str
        The start date in format "YYYY-MM-DD".
    end_date : str
        The end date in format "YYYY-MM-DD".

    Returns
    -------
    bool
        True if data exists, False otherwise.
    """
    df = df.loc[df["client_id"] == client_id]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    min_date = df["date"].min()
    max_date = df["date"].max()

    return (min_date <= pd.to_datetime(start_date)) and (max_date >= pd.to_datetime(end_date))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Testing tools")
    df = pd.read_csv("data/financial_data.csv")
    client_id = 1
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    logging.info(f"Client data exists: {check_client_data_exists(df, client_id)}")
    logging.info(f"Dates exist for client: {check_dates_exist_for_client(df, client_id, start_date, end_date)}")
    logging.info(f"Days in month: {get_days_in_month(2, 2024)}")
    logging.info("Tools tested")