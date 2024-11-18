from langchain_ollama import ChatOllama
import pandas as pd

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from copy import deepcopy
from langchain_core.runnables import chain
from langchain_core.globals import set_verbose, set_debug

from src.agent.tools import generate_report

import json

import logging

def run_agent(df: pd.DataFrame, client_id: int, input: str, verbose: bool=True) -> dict:
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
    if verbose:
        set_verbose(True)
        set_debug(True)

    tools = [generate_report]
    model = ChatOllama(model="llama3.2:1b", temperature=0)
    model = model.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''
                You are an AI agent that generates PDF reports using your available tools.
                For this, extract the start date and end date from the user input and call the `generate_report` tool.

                Remember the following:
                The mapping from month to number of days is as follows:
                - January: 31, February: 28 or 29, March: 31, April: 30, May: 31, June: 30, July: 31, August: 31, September: 30, October: 31, November: 30, December: 31.
                - For leap years, February has 29 days. (A leap year is a year that is divisible by 4, except for years that are divisible by 100 and not divisible by 400.)
                '''
            ),
            ("human", "{input}"),
        ]
    )

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             '''
    #             Call the `generate_report` tool with the information provided by the user. Do not provide instructions, just call the tool.
    #             '''
    #         ),
    #         ("human", "{input}"),
    #     ]
    # )

    @chain
    def inject_parameters(ai_msg):
        tool_calls = []
        for tool_call in ai_msg.tool_calls:
            tool_call_copy = deepcopy(tool_call)
            tool_call_copy["args"]["df"] = df
            tool_call_copy["args"]["client_id"] = client_id
            tool_calls.append(tool_call_copy)
        return tool_calls

    tool_map = {tool.name: tool for tool in tools}

    @chain
    def tool_router(tool_call):
        return tool_map[tool_call["name"]]

    full_chain: Runnable = prompt | model | inject_parameters | tool_router.map()
    response = full_chain.invoke(input)

    print(response)
    
    variables_dict = json.loads(response[0].content)
    return variables_dict

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Testing agent")

    sample_data = pd.read_csv(
        f"data/raw/transactions_data.csv", parse_dates=["date"]
    )
    input = "Create a pdf report for the fourth month of 2017"
    client_id = 122
    output = {
        "start_date": "2017-04-01",
        "end_date": "2017-04-30",
        "client_id": client_id,
        "create_report": True,
    }
    submitted_output = run_agent(
        input=input,
        client_id=client_id,
        df=sample_data.copy(deep=True),
    )