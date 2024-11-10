from langchain_ollama import ChatOllama
import pandas as pd

from langchain_core.prompts import ChatPromptTemplate

from tools import (earnings_and_expenses, expenses_summary, cash_flow_summary,
                   get_days_in_month, check_client_data_exists, check_dates_exist_for_client
                     )

def run_agent(df: pd.DataFrame, client_id: int, input: str) -> dict:
    """
    Create a simple AI Agent that generates PDF reports using the three functions from Task 2 (src/data/data_functions.py).
    The agent should generate a PDF report only if valid data is available for the specified client_id and date range.
    Using the data and visualizations from Task 2, the report should be informative and detailed.

    The agent should return a dictionary containing the start and end dates, the client_id, and whether the report was successfully created.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame  of the data to be used for the agent.
    client_id : int
        Id of the client making the request.
    input : str
        String with the client input for creating the report.


    Returns
    -------
    variables_dict : dict
        Dictionary of the variables of the query.
            {
                "start_date": "YYYY-MM-DD",
                "end_date" : "YYYY-MM-DD",
                "client_id": int,
                "create_report" : bool
            }

    """
    tools = [earnings_and_expenses, expenses_summary, cash_flow_summary, get_days_in_month,
             check_client_data_exists, check_dates_exist_for_client]
    model = ChatOllama(model="llama3.2:1b", temperature=0)
    model.bind_tools(tools)
    pdf_output_folder = "reports/"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''
                You are an AI agent. Your tasks are:
                1. Extract the start and end dates from the following user input:
                "{user_input}"
                2. Use the extracted dates, {client_id}, and {df} to generate a summary report in PDF format by invoking the necessary functions.
                '''
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | model
    chain.run(
        user_input=input,
        client_id=client_id,
        df=df
    )

    variables_dict = {
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "client_id": 0,
        "create_report": False,
    }

    return variables_dict


if __name__ == "__main__":
    ...
