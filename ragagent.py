import faiss
import openai
from openai import OpenAI
import numpy as np
import json
from dotenv import load_dotenv
import os
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from typing import Tuple, List, Set
import asyncio
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGChatAssistant:
    def __init__(self):
        """
        Initializes the RAG Chat Assistant with the FAISS index, metadata, and model configuration.
        """
        # self.config = config
        self.chat_history = []
        self.index, self.metadata = self.load_faiss_index()

    def load_faiss_index(self):
        """
        Load the FAISS index and corresponding metadata.
        """
        index = faiss.read_index("./data/db/vector_index.index")
        with open("./data/db/vector_index_metadata.json", "r") as f:
            metadata = json.load(f)
        return index, metadata

    def get_query_embedding(self, query:str) -> np.array:
        """
        Generate an embedding for a query using OpenAI.
        """
        client = OpenAI()
        response = client.embeddings.create(input=query, model="text-embedding-ada-002")
        return np.array(response.data[0].embedding, dtype=np.float32)

    def search_index(self, query_embedding:np.array, top_k=3) -> Tuple[List]:
        """
        Search the FAISS index for the most relevant chunks.
        """
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        results = []
        # sources = set()
        for idx in indices[0]:
            if idx != -1:  # Check for valid index
                results.append(self.metadata[idx])
                # sources.add(self.metadata[idx].get("original_file", "Unknown Source"))
        # return results, sources
        return results

    def generate_response(self, query:str, context:str) -> str:
        """
        Generate a response using OpenAI with chat history and retrieved context.
        """
        # Build a prompt with chat history and current context
        history_text = "\n\n".join(
            [f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in self.chat_history]
        )
        prompt = f"""
        You are an assistant with access to the following relevant information from a database:
        {context}

        Here is the chat history so far:
        {history_text}

        Based on the above information, answer the following question:
        {query}
        """
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Given the context presented, answer the question reliably and most accurately. Once the question is answered, print 'TERMINATE' so that we know you are done."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def chat(self, query:str, top_k=3):
        """
        Handle a single query by retrieving relevant chunks and generating a response.
        """
        # Generate query embedding
        query_embedding = self.get_query_embedding(query)

        # Search index for relevant chunks
        # results, sources = self.search_index(query_embedding, top_k=top_k)
        results = self.search_index(query_embedding, top_k=top_k)

        # Combine retrieved context
        context = "\n\n".join(
            [f"Source: {result['original_file']}\nText: {result['chunk']}" for result in results]
        )

        # Generate response
        response = self.generate_response(query, context)

        # Update chat history
        self.chat_history.append({"user": query, "assistant": response})

        # Add response + sources
        # full_response = response + "\n\n" + "The sources for this information are:\n" + "\n".join(f"- {source}" for source in sources)

        # return full_response, sources
        return response
    

def query_rag(query:str) -> str:
    """
    Executes a given query against a RAG and returns answer
    """
    assistant = RAGChatAssistant()
    response = assistant.chat(query)
    return response

query_rag_tool = FunctionTool(query_rag, description="An agent that can query a RAG and return result.")

PROMPT = """You are an agent designed to answer questions based on the information available in a Retrieval-Augmented Generation (RAG) database.
Your task is to first assess whether the question can be answered using the information within the RAG database. If the database does not contain relevant information, respond with a clear statement that you're unable to provide an answer.
For example, if the RAG database is focused on dogs and the question pertains to poems, simply reply, "I am not capable of answering this."
When the question is within the scope of the RAG database, provide a straightforward, factually accurate, and concise answer using only the context stored in the RAG database.
"""
run_rag_agent = AssistantAgent(
    name="DogReportAgent",
    model_client=OpenAIChatCompletionClient(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
    tools=[query_rag_tool],
    description="a tool that runs a query against a RAG db and returns the result",
    system_message=PROMPT#"You are a helpful AI assistant. Use the query and return the answer using the tool Solve tasks using your tools.",
)


async def main():
    termination = TextMentionTermination("TERMINATE")

    # Create team and run the group chat
    team = RoundRobinGroupChat([run_rag_agent],termination_condition=termination)

    # Run the group chat with a question
    stream = team.run_stream(task="What is this report about and which city is it based in?")
    
    # Output the result using Console
    await Console(stream)


if __name__ == "__main__":
    asyncio.run(main())
    # assistant = RAGChatAssistant()
    # print("Assistant: Hello! How can I help you?")
    # while True:
    #     user_query = input("You: ")

    #     # End the chat session
    #     if user_query.lower() in ["exit", "quit"]:
    #         print("Assistant: Goodbye!")
    #         break

    #     # Process the query and handle exceptions
    #     try:
    #         response, sources = assistant.chat(user_query)
    #         print(type(sources))
    #         print("\nAssistant:\n", response)
    #     except Exception as e:
    #         print("Assistant: Sorry, there was an error processing your request.")
    #         print(f"Error details: {e}")
    # response = query_rag(query="What is the ppt all about?")
    # print(response)