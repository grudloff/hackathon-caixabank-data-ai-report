import pandas as pd
from data_functions import get_transactions_dataset, get_clients_dataset, get_cards_dataset, get_mcc_codes
import json
import logging
logging.basicConfig(level=logging.INFO)

def question_1(cards_df):
    """
    Q1: - The `card_id` with the latest expiry date and the lowest credit limit amount.
    """
    logging.info("--------------------------------------------------")
    logging.info("Starting question 1...")
    logging.info("Question: The `card_id` with the latest expiry date and the lowest credit limit amount.")
    logging.info("Calculating results.")
    latest_expiry_date = cards_df["expires"].max()
    filtered_cards_df = cards_df[cards_df["expires"] == latest_expiry_date]
    lowest_credit_limit = filtered_cards_df["credit_limit"].min()
    card_id = filtered_cards_df["card_id"][filtered_cards_df["credit_limit"] == lowest_credit_limit].iloc[0]
    card_id = int(card_id)
    logging.info(f"Card ID: {card_id}, Expiry Date: {latest_expiry_date}, Credit Limit: {lowest_credit_limit}")
    return card_id
    


def question_2(client_df):
    """
    Q2: - The `client_id` that will retire within a year that has the lowest credit score and highest debt.
    """
    logging.info("--------------------------------------------------")
    logging.info("Starting question 2...")
    logging.info("Question: The `client_id` that will retire within a year that has the lowest credit score and highest debt.")
    logging.info("Filtering dataframes.")
    filtered_client_df = client_df[client_df["retirement_age"] - client_df["current_age"] <= 1]
    logging.info("Calculating results.")
    lowest_credit_score = filtered_client_df["credit_score"].min()
    filtered_client_df = filtered_client_df[filtered_client_df["credit_score"] == lowest_credit_score]
    highest_debt = filtered_client_df["total_debt"].max()
    mask = (filtered_client_df["credit_score"] == lowest_credit_score) & (filtered_client_df["total_debt"] == highest_debt)
    client_id = filtered_client_df["client_id"][mask].iloc[0]
    client_id = int(client_id)
    logging.info(f"Client ID: {client_id}, Credit Score: {lowest_credit_score}, Total Debt: {highest_debt}")
    return client_id


def question_3(transactions_df):
    """
    Q3: - The `transaction_id` of an Online purchase on a 31st of December with the highest absolute amount (either earnings or expenses).
    """
    logging.info("--------------------------------------------------")
    logging.info("Starting question 3...")
    logging.info("Question: The `transaction_id` of an Online purchase on a 31st of December with the highest absolute amount (either earnings or expenses).")
    logging.info("Filtering dataframes.")
    filtered_transactions_df = transactions_df[(transactions_df["date"].dt.day == 31) & (transactions_df["date"].dt.month == 12)]
    logging.info("Calculating results.")
    highest_amount = filtered_transactions_df["amount"].abs().max()
    transaction_id = filtered_transactions_df["transaction_id"][filtered_transactions_df["amount"].abs() == highest_amount].iloc[0]
    transaction_id = int(transaction_id)
    logging.info(f"Transaction ID: {transaction_id}, Amount: {highest_amount}")
    return transaction_id


def question_4(client_df, cards_df, transactions_df):
    """
    Q4: - Which client over the age of 40 made the most transactions with a Visa card in February 2016?
    Please return the `client_id`, the `card_id` involved, and the total number of transactions.
    """
    logging.info("--------------------------------------------------")
    logging.info("Starting question 4...")
    logging.info("Question: Which client over the age of 40 made the most transactions with a Visa card in February 2016?")
    logging.info("Filtering dataframes.")
    filtered_client_df = client_df[client_df["current_age"] > 40]
    filtered_cards_df = cards_df[cards_df["card_brand"] == "Visa"]
    filtered_transactions_df = transactions_df[(transactions_df["date"].dt.month == 2) & (transactions_df["date"].dt.year == 2016)]
    logging.info("Merging dataframes.")
    transactions_df = filtered_transactions_df[filtered_transactions_df["card_id"].isin(filtered_cards_df["card_id"].unique())]
    transactions_df = transactions_df[transactions_df["client_id"].isin(filtered_client_df["client_id"].unique())]
    logging.info("Calculating results.")
    transactions_per_card = transactions_df["card_id"].value_counts()
    card_id = transactions_per_card.idxmax()
    client_id = transactions_df[transactions_df["card_id"] == card_id]["client_id"].iloc[0]
    total_transactions = transactions_per_card.max()
    client_id = int(client_id)
    card_id = int(card_id)
    total_transactions = int(total_transactions)
    logging.info(f"Client ID: {client_id}, Card ID: {card_id}, Total Transactions: {total_transactions}")
    return client_id, card_id, total_transactions

def write_responses(response_1, response_2, response_3, response_4):
    """
    Write the responses to a json file.
    """
    responses = {
        "target": {
            "query_1": {
                "card_id": response_1
            },
            "query_2": {
                "client_id": response_2
            },
            "query_3": {
                "transaction_id": response_3
            },
            "query_4": {
                "client_id": response_4[0],
                "card_id": response_4[1],
                "number_transactions": response_4[2]
            }
        }
    }
    with open("predictions/predictions_1.json", "w") as file:
        json.dump(responses, file, indent=4)

if __name__ == "__main__":
    cards_df = get_cards_dataset()
    client_df = get_clients_dataset()
    transactions_df = get_transactions_dataset()
    
    response_1 = question_1(cards_df)
    response_2 = question_2(client_df)
    response_3 = question_3(transactions_df)
    response_4 = question_4(client_df, cards_df, transactions_df)

    write_responses(response_1, response_2, response_3, response_4)