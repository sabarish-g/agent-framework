# from autogen_agentchat.agents import AssistantAgent
# from autogen_agentchat.teams import RoundRobinGroupChat
# from autogen_agentchat.ui import Console
# from autogen_ext.models.openai import OpenAIChatCompletionClient
# import asyncio
from dotenv import load_dotenv
import os 

load_dotenv()
# # Define a tool
# async def get_weather(city: str) -> str:
#     return f"The weather in {city} is 73 degrees and Sunny."


# async def main() -> None:
#     # Define an agent
#     weather_agent = AssistantAgent(
#         name="weather_agent",
#         model_client=OpenAIChatCompletionClient(
#             model="gpt-4o",
#             api_key=os.getenv("OPENAI_API_KEY")
#         ),
#         tools=[get_weather],
#     )

#     # Define a team with a single agent and maximum auto-gen turns of 1.
#     agent_team = RoundRobinGroupChat([weather_agent], max_turns=1)

#     while True:
#         # Get user input from the console.
#         user_input = input("Enter a message (type 'exit' to leave): ")
#         if user_input.strip().lower() == "exit":
#             break
#         # Run the team and stream messages to the console.
#         stream = agent_team.run_stream(task=user_input)
#         await Console(stream)


# # NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o", seed=42, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. Write all code in python. Reply only 'TERMINATE' if the task is done.",
        model_client=model_client,
    )

    code_executor = CodeExecutorAgent(
        name="code_executor",
        code_executor=LocalCommandLineCodeExecutor(work_dir="coding"),
    )

    # The termination condition is a combination of text termination and max message termination, either of which will cause the chat to terminate.
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)

    # The group chat will alternate between the assistant and the code executor.
    group_chat = RoundRobinGroupChat([assistant, code_executor], termination_condition=termination)

    # `run_stream` returns an async generator to stream the intermediate messages.
    stream = group_chat.run_stream(task="Write a python script to print 'Hello, world!'")
    # `Console` is a simple UI to display the stream.
    await Console(stream)

asyncio.run(main())