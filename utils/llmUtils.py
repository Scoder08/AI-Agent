import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_openai_tools_agent, create_tool_calling_agent
import logging
import traceback
from langchain_core.messages import HumanMessage

class LLM():
        LLM_OPEN_AI = "openai"
        LLM_GOOGLE = "google"

logger = logging.getLogger(__name__)
llmType = "openai"

def getLLM(type : str, name: str = None):
    match type:
        case LLM.LLM_OPEN_AI:
            return ChatOpenAI(name=name, model= "gpt-4o")
        case LLM.LLM_GOOGLE:
            return ChatGoogleGenerativeAI(
            name = name,
            model="gemini-1.5-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            verbose = True,
        )


def getLlmAgent(tools, llm, prompt):
    agent = None
    if llmType == "openai":
        logger.info("openai agent")
        agent = create_openai_tools_agent(
            tools=tools,
            llm=getLLM(llmType),
            prompt=prompt
        )
    else:
        logger.info("not a openai agent")

    if llmType == "google":
        logger.info("google agent")
        agent = create_tool_calling_agent(
            tools=tools,
            llm=getLLM(llmType),
            prompt=prompt
        )
    else:
        logger.info("not a google agent")

    return agent

def approximate_token_count(text):
    words = text.split()
    return len(words) + sum(len(word) for word in words) // 4

def filter_messages(messages: list, max_tokens: int = 59000) -> list:
    filtered_messages = []
    total_tokens = 0

    # Loop backward through the messages to get the last ones first
    for message in reversed(messages):
        if type(message) == str:
            message_tokens = approximate_token_count(message)
        else:
            message_tokens = approximate_token_count(message.content)
        if total_tokens + message_tokens > max_tokens:
            break
        filtered_messages.append(message)
        total_tokens += message_tokens

    # Return the filtered messages in their original order
    return list(reversed(filtered_messages))

