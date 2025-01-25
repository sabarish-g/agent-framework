from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
import asyncio
from sqlagent import run_query_agent, generate_query_agent
from ragagent import run_rag_agent
from autogen_agentchat.conditions import TextMentionTermination
import sqlite3
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
import sqlite3
from dotenv import load_dotenv
import os
import asyncio
load_dotenv()


openai_key = os.getenv("OPEN_API_KEY")


async def main():
    termination = TextMentionTermination("TERMINATE")
    # Create team and run the group chat
    team = SelectorGroupChat([run_rag_agent, run_query_agent, generate_query_agent], termination_condition=termination, 
                             model_client=OpenAIChatCompletionClient(model="gpt-4o"))

    # Run the group chat with a question
    stream = team.run_stream(task="How many apple phones are in stock? Use the sql generation and the execution agents and not the rag agent.")#
    # Output the result using Console
    await Console(stream)


if __name__ == "__main__":
    asyncio.run(main())