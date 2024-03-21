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
import datetime


load_dotenv(find_dotenv())
db = SQLDatabase.from_uri("sqlite:///TriggTest.db")
now = datetime.datetime.now()

#TODO: Run test to check if the querys are correct when there's data to test with.
examples = [
    {"input": "List all coupons.", "query": "SELECT * FROM coupons;"},
    {"input": "List all customers.", "query": "SELECT * FROM customers;"},
    {"input": "List all campaigns.", "query": "SELECT * FROM bonusCampaign;"},

    {
        "input": "Which coupons are active?",
        "query": "SELECT name FROM coupons WHERE validFrom <= datetime(now) =< validTo;"
    },
    {
        "input": "Which campaign is the most popular?",
        "query": 'SELECT TOP 1 bc.id, count(*) FROM couponRedeem cr INNER JOIN coupons c ON cr.idCoupon = c.id INNER JOIN bonusCampaign bc ON c.idCampaign = bc.id GROUP BY bc.id ORDER BY count(*) DESC'
    },
    {
        "input": "How many jubilee offer coupons has been redeemed?",
        "query": "SELECT COUNT(*) FROM couponRedeem INNER JOIN coupons ON couponRedeem.idCoupon = coupons.id WHERE coupons.name = 'Jubilee offer';"
    },
    {
        "input": "How many customers has redeemed the jubilee offer coupon?",
        "query": "SELECT COUNT(*) FROM couponRedeem INNER JOIN customerCoupon ON customerCoupon.idCoupon = couponReedem.idCoupon WHERE customerCoupon.idCoupon IN (SELECT name FROM coupons WHERE name ='Jubilee offer');",
    },
    {
        "input": "how long is jubilee offer coupon active?",
        "query": "SELECT julianday(validTo) - julianday(validFrom) AS Duration FROM coupons WHERE name = 'jubilee offer';",
    },
    {
        "input": "How many coupons are there?",
        "query": 'SELECT COUNT(*) FROM "coupons"',
    },
]

# Consider changing to Milvus vector database, if the examples are getting very large.
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


def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


coupons = query_as_list(db, "SELECT name FROM coupons")
campaigns = query_as_list(db, "SELECT name FROM bonusCampaign")

vector_db = Annoy.from_texts(campaigns + coupons, OpenAIEmbeddings())
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
valid proper nouns. Use the noun most similar to the search."""
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)


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


@cl.on_message
async def on_message(message: cl.Message):
    # Run model
    msg = cl.Message(content=message.content)

    agent = cl.user_session.get("runnable")  # type: Runnable

    response = agent.invoke(input=msg.content)

    print(response)

    response_str = json.dumps(response.get("output"), ensure_ascii=False).encode('utf8')[1:-1]

    # Send a response back to the user
    await cl.Message(
        content=response_str.decode(),
    ).send()

# chainlit run app.py -w
