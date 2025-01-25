from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
import duckdb
import pandas as pd
from dotenv import load_dotenv
import os
import asyncio
load_dotenv()
openai_key = os.getenv("OPEN_API_KEY")
termination = TextMentionTermination("TERMINATE")
def query_db(query:str, df_path:str="./data/mobile_stock.csv"):
    """
    Executes a given SQL query on the specified SQLite database and returns the rows.
    """
    # Query with DuckDB
    # query = "SELECT * FROM df WHERE Available_Quantity > 20"
    df = pd.read_csv(df_path)
    # Execute the query directly on the pandas DataFrame
    result = duckdb.query(query).df()
    return result

query_db_tool = FunctionTool(query_db, description="An agent that can query a database and return result.")

run_query_agent = AssistantAgent(
    name="SQLQueryAgent",
    model_client=OpenAIChatCompletionClient(model="gpt-4o", api_key=openai_key),
    tools=[query_db_tool],
    description="a tool that runs a query against a sqlite3 db and returns the result",
    system_message="""You are a helpful AI assistant. 
    Use the query and return the answer using the tool Solve tasks using your tools.
    Once you are done, return the final answer and type 'TERMINATE'. """,
)

PROMPT = """You are a SQL query generator for a database. 
Your task is to generate an SQL query based on the following schema. 
Only use the listed columns and tables in your query. 
Do not use any columns or tables that are not provided below. 
Please ensure the generated SQL query is syntactically correct and meets the requirements of the given question.
Below is the Schema of the available tables to generate the SQL queries:

CREATE TABLE `df` (
  `Product_id` int NOT NULL,
  `Product_Name` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `Available_Quantity` int DEFAULT NULL,
  `Mobile_Specs` text COLLATE utf8mb4_unicode_ci,
  PRIMARY KEY (`Product_id`),
  FULLTEXT KEY `mobile_stock_Product_Name_IDX` (`Product_Name`,`Mobile_Specs`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

Only use the columns `Product_id`, `Product_Name`, `Available_Quantity`, and `Mobile_Specs` in the query, and ensure that your query:
1. Correctly handles filtering, aggregation, sorting, and joins if necessary (based on the question).
2. use the correct table name.
3. Does not include any operations on columns or tables that are not listed in the schema above.
4. Returns a result that directly answers the given question. 

For the given question, generate a single SQL query that will return the correct answer based on the schema provided.
Once you have genereated the SQL query, hand it over to the run_query_agent.
"""

generate_query_agent = AssistantAgent(
    name="GenerateQueryAgent",
    model_client=OpenAIChatCompletionClient(model="gpt-4o", api_key=openai_key),
    description="a tool that runs a query against a sqlite3 db and returns the result",
    system_message=PROMPT,
)

async def main():
    # Create team and run the group chat
    team = RoundRobinGroupChat([generate_query_agent, run_query_agent], termination_condition=termination)

    # Run the group chat with a question
    stream = team.run_stream(task="what are the different brands(not models) of phones and how many of them do we have in stock?")
    
    # Output the result using Console
    await Console(stream)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())