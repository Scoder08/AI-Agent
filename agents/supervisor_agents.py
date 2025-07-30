from typing import Annotated
from agents.sam import sam
from agents.satwik import satwik
from agents.preview import preview  
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
import operator
from typing import Sequence
from typing_extensions import TypedDict
from utils.llmUtils import getLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from tools.tools_list import tools_list
from langchain_core.runnables.config import RunnableConfig
from utils.datetime_utils import get_current_time_with_offset
from utils.llmUtils import filter_messages
from langchain.tools import tool   

class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]


async def SAM(session_id, query, config: RunnableConfig):
    try:
        result = await sam.ainvoke({"messages": [f"session_id: {session_id}\n"] + [HumanMessage(content=query)]}, config=config)
        content = result["messages"][-1].content
    except Exception as e:
        content = "I am sorry I could not find the information you are looking for. Please try again later."
    return content


async def SATWIK(session_id, query, config: RunnableConfig):
    try:
        result = await satwik.ainvoke({"messages": [f"session_id: {session_id}\n"] + [HumanMessage(content=query)]}, config=config)
        content = result["messages"][-1].content
    except Exception as e:
        content = "I am sorry I could not find the information you are looking for. Please try again later."
    return content

async def PREVIEW(session_id, query, config: RunnableConfig):
    try:
        result = await preview.ainvoke({"messages": [f"session_id: {session_id}\n"] + [HumanMessage(content=query)]}, config=config)
        content = result["messages"][-1].content
    except Exception as e:
        content = "I am sorry I could not find the information you are looking for. Please try again later."
    return content


subordinate_agents = {
    "SAM": SAM,
    "SATWIK" : SATWIK
}

@tool("SAM", return_direct=True)      # wrapper for function-calling
async def sam_tool(session_id: str, query: str, config: RunnableConfig):
    """
    Answer *operational* order-level questions in natural language.

    Typical uses ▸ current status of an order or item, refund details (RRN, UTR,
    amount, bank), delivery window, cancel/return eligibility, or any other
    real-time customer-service information pulled from the order-item payload.
    """
    return await SAM(session_id, query, config)

@tool("SATWIK", return_direct=True)
async def satwik_tool(query: str, session_id: str = '', config: RunnableConfig = None):
    """
    Generate production-ready **ClickHouse SQL** for order-item analytics.

    Typical uses ▸ counts, percentages, cohort metrics, cancellation / return /
    refund analysis, or any ad-hoc reporting query that must follow the
    `analytics.order_items_view` schema and built-in business rules.
    """
    return await SATWIK(session_id, query, config)

@tool("PREVIEW", return_direct=True)
async def preview_tool(query: str, session_id: str = '', config: RunnableConfig = None):
    """
    **Code-Review Assistant**

    Pass a GitHub pull-request URL (e.g.  
    `https://github.com/org/repo/pull/123`).  
    The agent will download the diff, inspect the changed files, and return
    a markdown review that covers:

    • design / architecture  
    • readability & style issues  
    • possible bugs and edge-cases  
    • duplicated logic or functions already present in the repo  
    • inline ```suggestion``` blocks with improved code where helpful  

    Results come back directly in Slack; the bot never writes to GitHub.
    """
    return await PREVIEW(session_id, query, config)

subordinate_tools = [sam_tool, satwik_tool, preview_tool]

system_prompt = (
    """
    You are the supervisor of the Newme AI agent swarm — a team of specialized agents that collaboratively assist users by answering their queries. Your current active team members are:

    • SAM — an expert in handling all queries related to:
        - Orders
        - Order items
        - Order statuses
        - Refunds and refund statuses
        - Delivery and cancellation information

    • SATWIK — an expert in generating highly accurate and optimized SQL queries for ClickHouse databases.
        - Already has access to the schema for `analytics.order_items_view`in schema provided dictionary.
        - You can delegate any query involving order item analytics, statuses, cancellations, or refund metrics to SATWIK using tables schema.

    • PREVIEW — an expert PR reviewer. Pass any GitHub pull-request URL
       (e.g. https://github.com/org/repo/pull/123) to PREVIEW. It will
       fetch the diff itself — no need to supply code.

    As the supervisor, your job is to:
    - Professionally interpret the user’s query.
    - Determine which specialized agent (SAM or SATWIK or PREVIEW) is best suited to handle the request.
    - If delegation is required:
        - For SAM: pass a minimal instruction with required IDs (e.g., order ID, item ID).
        - For SATWIK: pass the user’s query along with the table name to use (e.g., `analytics.order_items_view`).
    - If the query is unrelated to either agent's scope, politely decline.

    Guidelines:
    1. Do not ask for the table schema again if it is already known (like `analytics.order_items_view`).
    2. For SQL requests involving metrics like cancellations, returns, quantities, status counts, refunds, or percentages — SATWIK should be used.
    3. If a user asks for operational status, live tracking, or action on a specific order — delegate to SAM.
    4. Never fabricate data. Request missing identifiers (order ID, item ID) if needed.
    5. Always respond clearly and concisely, without unnecessary pleasantries.
    6. Delegate GitHub PR review tasks to PREVIEW.
    7. While replying make sure to make it clearly and elegantely readable on slack. no need to add extra * in text like **hi**

    Remove **...** styling
    Present responses in clean, readable plain text
    Use emojis or line breaks (optional) for visual clarity in Slack

    Your goal is to ensure that the user’s query is routed to the correct agent, resulting in an accurate and context-aware resolution.
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial()

llm = getLLM("openai", "SUPERVISOR")

# tool_node = ToolNode(tools_list["SUPERVISOR"]+list(subordinate_agents.values()))
tool_node = ToolNode(subordinate_tools) 

supervisor_chain = (
    prompt
    | llm.bind_tools(subordinate_tools)
    # | (lambda x: x.dict()) # convert pydantic to dict for graph update
)

async def invoke(state: dict, config: RunnableConfig):
    state["messages"] = filter_messages(state["messages"])
    state["messages"].append(SystemMessage(
                                content=f"Today is {get_current_time_with_offset(config):%d-%m-%Y %H:%M:%S}. "
                                f"TZ={config.get('configurable', {}).get('timezone', 'Asia/Kolkata')} "
                                f"offset={config.get('configurable', {}).get('timezone_offset', 330)}m."
                            ))
    result = await supervisor_chain.ainvoke(state, config)
    return {"messages": [result]}

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there are tool_calls, we continue
    else:
        return "continue"

workflow = StateGraph(AgentState)

workflow.add_node("SUPERVISOR", invoke)
workflow.add_node("ACTION", tool_node)
workflow.add_edge("ACTION", "SUPERVISOR")

workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "SUPERVISOR",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # END is a special node marking that the graph should finish.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "ACTION",
        # Otherwise we finish.
        "end": END
    }
)

# Finally, add entrypoint
workflow.add_edge(START, "SUPERVISOR")


async def get_supervisor():
    # DB_URI = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}?sslmode=disable"
    # logger.info(f"Connecting to database at {DB_URI}")
    # connection_kwargs = {
    #     "autocommit": True,
    #     "prepare_threshold": 0,
    # }
    # conn = await AsyncConnection.connect(DB_URI, **connection_kwargs)
    # memory = AsyncPostgresSaver(conn)
    # await memory.setup()
    supervisor = workflow.compile(checkpointer=MemorySaver())
    return supervisor
