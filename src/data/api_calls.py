# file for implementing api call functions
import requests
import pandas as pd

def get_client_data(client_id: int) -> dict:
    """
    Get the client data from the API.

    Parameters
    ----------
    client_id : int
        Id of the client.

    Returns
    -------
    dict
        Dictionary with the client data.
    """
    api_endpoint = "https://faas-lon1-917a94a7.doserverless.co/api/v1/web/fn-a1f52b59-3551-477f-b8f3-de612fbf2769/default/clients-data"
    response = requests.get(api_endpoint, params={"client_id": client_id})
    return {client_id : response.json()["values"]}

def get_cards_data(client_id: int) -> dict:
    """
    Get the cards data from the API.

    Parameters
    ----------
    client_id : int
        Id of the client.

    Returns
    -------
    dict
        Dictionary with the cards data.
    """
    api_endpoint = "https://faas-lon1-917a94a7.doserverless.co/api/v1/web/fn-a1f52b59-3551-477f-b8f3-de612fbf2769/default/cards-data"
    response = requests.get(api_endpoint, params={"client_id": client_id})
    return response.json()["values"]

if __name__ == "__main__":
    ...
