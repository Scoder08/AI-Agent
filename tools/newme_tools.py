import traceback
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
import requests
from constants import DATA_CUTOFF_LIMIT
import logging
import json

base_url = "https://internaltoolsapi.newme.asia/kapture/fetch/"
auth_token = "<NEWME AUTH TOKEN>"
logger = logging.getLogger(__name__)

class GetOrderDetailsPromptInput(BaseModel):
    order_id: int = Field("order_id")
    order_item_id: int = Field("order_item_id")
    user_prompt: str = Field(
        ...,
        description="A pricise single sentence question describing the data user is looking for exactly. Mention the exact data point user is asking for. Precisely ask for count if user want to get count."
    )

@tool("get_order_item_details_tool", args_schema=GetOrderDetailsPromptInput)
def get_order_item_details_tool(order_id : int, order_item_id : int, user_prompt: str):
    """
    Retrieves detailed information about a specific order based on the provided order ID, item ID, 
    and a natural language prompt describing the required data.

    Args:
        order_id (int): The unique identifier for the order.
        order_item_id (int): The unique identifier for the order item.
        user_prompt (str): A precise natural language description of the specific data the user wants to retrieve.

    Returns:
        dict: A dictionary containing the retrieved order details. If the total number of entries exceeds 
              the defined cutoff limit, the result will be truncated and include a message indicating this.

    Raises:
        Exception: If an error occurs while retrieving the order details.
    """
    try:
        data = get_item_details(order_id, order_item_id, user_prompt)
        result = {'data' : data, 'message' : "I have retrieved your items details succesfully"}
        return result

    except Exception as e:
        logger.error(traceback.format_exc())
        return f"Error calling get_order_details_tool: {e}"


def get_item_details(order_id, order_item_id, user_prompt):
    url = base_url + 'item/details'
    payload = json.dumps({
        "is_internal": True,
        "page_number": "1",
        "order_id": order_id,
        "order_item_id": order_item_id
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {auth_token}',
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    data = response.json()
    logger.info(data)
    return data

def get_customer_details():
    pass

def get_item_details_tool():
    pass

        
def filter_docs_to_limit(docs, limit):
    """Filter the documents to the specified limit."""
    new_docs = {}
    num_data = 0
    for key, value in docs.items():
        if type(key) == int and num_data < limit:
            new_docs[key] = value
            num_data += 1
        else:
            new_docs[key] = value
    return new_docs