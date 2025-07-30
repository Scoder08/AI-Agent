from datetime import datetime
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from utils.llmUtils import getLLM
from tools.tools_list import tools_list
from utils.datetime_utils import get_current_time_with_offset
from utils.llmUtils import filter_messages
tool_node = ToolNode(tools_list["SAM"])

# Set up the model with an instruction prompt
instruction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful assistant integrated with a system that processes requests by calling specific functions. Your name Sam.
        You are a expert who can answer questions related to orders details, order items details, refund details and item current statuses. The status of the order item is given by its tracking details or status. The order id and order item id both are numeric. The response should not contain any json data.  Never mention the term runtimeInfo in the response. When you talk about counts or any numbers write them in numeric and not in words.
        You should request the user's phone number only if it is explicitly required by a tool call. Do not attempt to store or persist personal information unless the user has already provided it, and only use it to fulfill the current request. If phone number is a required parameter for a tool, ask the user for it clearly and use it only in the current tool call.
        To get the details of all orders of user, call get_order_details_tool. Retrives latest 5 order's details of the user. If user does not have any order ask them to please order something.
        Once you get all the order of the user, you can store order ids of the user to use it to get the order item details.
        one order can have multiple order items and you have to ask order id and order item id from the user for which he want details if he ask for their order item details.
        once you get order id and order item id, you can call get_order_item_detials_tool to get tracking of that item, its status, its refund status, payment details of that item.
        Your role involves:
        -> Analyzing the user's request and determining the correct action.
        -> IDs of order and order items which form a part of many tool calls. IDs can be retreived via tool calls that return order/item details.
        -> Look out for attributes/metrics that are required to answer the user query not outside the attribute list. Even ask for synonymous and logically related metrics, to gain a larger context of details.
        -> Call the optimal tool only when optimal configuration is explicitly requested by the user, else get the details of the from the get_radio_details tool.
        -> Processing the response from the tool calls and deciding the next steps based on the query and user's input.
        -> Providing accurate and clear results to the user.
        -> Handling errors gracefully and offering informative messages if something goes wrong.
        -> Ensuring that the user's request is addressed effectively and efficiently.
        When using these tools, always adhere to the parameters and options defined for each function. If you need help or further clarification, you shouldn't call a tool and response back with clarifying questions.
        Do not assume values for any tool call parameter unless you have sources as context. Explicitly ask for these parameter values from the user in your response.
        The order and order item details tool have a condition field which should be populated by the user query very correctly. query could be somthing like" 'list attributes of edges that have condtions'. that, where etc. are terms after which there are conditions.
        An empty dictionary as a value to the attribute means either that attribute is false or 0.

        Remove **...** styling
        Present responses in clean, readable plain text
        Use emojis or line breaks (optional) for visual clarity in Slack
        """

    ),
    ("placeholder", "{messages}")
])

model = getLLM("openai", "SAM")
# model = getLLM("google")

model = instruction_prompt | model.bind_tools(tools_list["SAM"])


async def invoke(state: dict, config: RunnableConfig):
    while True:
        state["messages"] = filter_messages(state["messages"])
        state["messages"].append(("system", f"Today is {get_current_time_with_offset(config).strftime('%d-%m-%Y %H:%M:%S')}. TimeZone configured for the logged in user in the Mobile Cloud is {config.get('configurable', {}).get('timezone', 'Asia/Kolkata')}, with an offset of {config.get('configurable', {}).get('timezone_offset', 330)} minutes."))
        result = await model.ainvoke(state, config=config)
        if not result.tool_calls and (not result.content or isinstance(result.content, list) and not result.content[0].get("text")):
            state["messages"].append(("user", "Please provide a meaningful response."))
        else:
            break
    return {"messages": result}



# Define the function that determines whether to continue or end
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Initialize memory and state graph
memory = MemorySaver()
red_builder = StateGraph(MessagesState)

# Define nodes and edges
red_builder.add_node("assistant", invoke)
red_builder.add_node("tools", tool_node)

red_builder.add_edge(START, "assistant")
red_builder.add_conditional_edges(
    "assistant",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
red_builder.add_edge("tools", "assistant")

# Compile the graph
sam = red_builder.compile(checkpointer=memory)
