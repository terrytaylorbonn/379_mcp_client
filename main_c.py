# mcp_client/main_c.py GPT17

import os
import json
import requests
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import openai

app = FastAPI()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")  # Loaded from env var
OPENAI_MODEL = "gpt-3.5-turbo"

openai.api_key = os.getenv("OPENAI_API_KEY")

class QuestionContext(BaseModel):
    question: str = Field(..., description="User's question")

class AnswerOutput(BaseModel):
    answer: str = Field(..., description="Answer to user's question")

def fetch_tool_metadata():
    resp = requests.get(f"{MCP_SERVER_URL}/metadata")
    return resp.json()

tool_metadata = fetch_tool_metadata()

def llm_decide_tool_use(question: str, tool_metadata: dict) -> dict:
    tool_description = (
        f"Tool Name: {tool_metadata['tool_name']}\n"
        f"Description: {tool_metadata['description']}\n"
        f"Input Schema: {json.dumps(tool_metadata['input_schema'])}"
    )

    prompt = f"""
You are an AI assistant. A user asked the question: "{question}".

You have access to the following tool:

{tool_description}

Decide if using the tool is necessary to answer the user's question.

If YES, return:
use_tool: true
tool_input: <country name>

If NO, return:
use_tool: false
tool_input: null

Respond in JSON format.
"""

    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response["choices"][0]["message"]["content"]

    try:
        decision = json.loads(content)
        return decision
    except Exception:
        return {"use_tool": False, "tool_input": None}

@app.get("/")
def root():
    return {
        "message": "MCP Client — Real LLM + Tool Selection",
        "tool_metadata": tool_metadata
    }

@app.get("/ask")
def ask(question: str = Query(..., description="User question")):
    context = QuestionContext(question=question)

    decision = llm_decide_tool_use(context.question, tool_metadata)

    if decision["use_tool"]:
        response = requests.post(
            f"{MCP_SERVER_URL}{tool_metadata['endpoint']}",
            json={"country": decision["tool_input"]}
        )
        data = response.json()
        capital = data.get("capital", "Unknown")
        answer = f"The capital of {decision['tool_input']} is {capital}."
    else:
        fallback_response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": f"Answer this question directly: {context.question}"}],
            temperature=0
        )
        answer = fallback_response["choices"][0]["message"]["content"].strip()

    return AnswerOutput(answer=answer)
