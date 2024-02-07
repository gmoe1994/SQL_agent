from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import Annoy
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_community.utilities import SQLDatabase
import chainlit as cl
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
import ast
import re
import json

load_dotenv(find_dotenv())
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

examples = [
    {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
    {
        "input": "Find all albums for the artist 'AC/DC'.",
        "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
    },
    {
        "input": "List all tracks in the 'Rock' genre.",
        "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
    },
    {
        "input": "Find the total duration of all tracks.",
        "query": "SELECT SUM(Milliseconds) FROM Track;",
    },
    {
        "input": "List all customers from Canada.",
        "query": "SELECT * FROM Customer WHERE Country = 'Canada';",
    },
    {
        "input": "How many tracks are there in the album with ID 5?",
        "query": "SELECT COUNT(*) FROM Track WHERE AlbumId = 5;",
    },
    {
        "input": "Find the total number of invoices.",
        "query": "SELECT COUNT(*) FROM Invoice;",
    },
    {
        "input": "List all tracks that are longer than 5 minutes.",
        "query": "SELECT * FROM Track WHERE Milliseconds > 300000;",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    },
    {
        "input": "Which albums are from the year 2000?",
        "query": "SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';",
    },
    {
        "input": "How many employees are there",
        "query": 'SELECT COUNT(*) FROM "Employee"',
    },
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Annoy,
    k=5,
    input_keys=["input"],
)

system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect", "top_k"],
    prefix=system_prefix,
    suffix="",
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# def query_as_list(db, query):
#     res = db.run(query)
#     res = [el for sub in ast.literal_eval(res) for el in sub if el]
#     res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
#     return list(set(res))
#
#
# artists = query_as_list(db, "SELECT Name FROM Artist")
# albums = query_as_list(db, "SELECT Title FROM Album")
#
# vector_db = Annoy.from_texts(artists + albums, OpenAIEmbeddings())
# retriever = vector_db.as_retriever(search_kwargs={"k": 5})
# description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
# valid proper nouns. Use the noun most similar to the search."""
# retriever_tool = create_retriever_tool(
#     retriever,
#     name="search_proper_nouns",
#     description=description,
# )

@cl.on_chat_start
async def on_chat_start():
    # model = ChatOpenAI(streaming=True)

    agent = create_sql_agent(
        llm=llm,
        db=db,
        prompt=full_prompt,
        verbose=True,
        agent_type="openai-tools",
    )

    cl.user_session.set("runnable", agent)


# @cl.on_message
# async def on_message(message: cl.Message):
#     runnable = cl.user_session.get("runnable")  # type: Runnable
#
#     msg = cl.Message(content="")
#
#     async for chunk in runnable.astream(
#             {
#                 "input": message.content
#             },
#             config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
#     ):
#         await msg.stream_token(chunk)
#
#     await msg.send()

@cl.on_message
async def on_message(message: cl.Message):
    # Run model
    msg = cl.Message(content=message.content)

    agent = cl.user_session.get("runnable")  # type: Runnable

    response = agent.invoke(input=msg.content)

    response_str = json.dumps(response.get("output"))[1:-1]

    # Send a response back to the user
    await cl.Message(
        content=response_str,
    ).send()
