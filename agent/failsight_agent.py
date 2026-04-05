"""
LangGraph AI Agent: RAG chatbot + auto failure report generator
"""
from config import OPENAI_API_KEY

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from typing import TypedDict, Annotated, List
import operator

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import chromadb

from config import (
    DATA_PROCESSED_DIR, VECTORSTORE_DIR, MODELS_DIR,
    RUL_CRITICAL_THRESHOLD, RUL_WARNING_THRESHOLD, LLM_MODEL
)
from models.train_rul import predict_rul


# ── State ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    retrieved_context: str
    rul_data: str
    report: str
    response: str
    messages: Annotated[List[str], operator.add]


# ── Vector Store Builder ───────────────────────────────────────────────────
def build_vectorstore(df: pd.DataFrame):
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    collection = client.get_or_create_collection("fleet")
    
    latest = df.sort_values("cycle").groupby("unit").last().reset_index()
    preds = predict_rul(latest)
    latest["predicted_rul"] = preds.clip(min=0)
    
    docs, ids = [], []
    for _, row in latest.iterrows():
        status = "CRITICAL" if row["predicted_rul"] <= RUL_CRITICAL_THRESHOLD else "WARNING" if row["predicted_rul"] <= RUL_WARNING_THRESHOLD else "HEALTHY"
        text = f"Unit {int(row['unit'])} | Cycle: {int(row['cycle'])} | RUL: {row['predicted_rul']:.1f} | Status: {status}"
        docs.append(text)
        ids.append(f"unit_{int(row['unit'])}")
    
    collection.upsert(documents=docs, ids=ids)
    print(f"Vectorstore built with {len(docs)} documents.")
    


def load_vectorstore():
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    return client.get_or_create_collection("fleet")


# ── LLM ───────────────────────────────────────────────────────────────────

def get_llm():
    from config import OPENAI_API_KEY
    return ChatOpenAI(model=LLM_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)


# ── Nodes ──────────────────────────────────────────────────────────────────
def retrieve_context(state: AgentState) -> AgentState:
    try:
        collection = load_vectorstore()
        results = collection.query(query_texts=[state["query"]], n_results=5)
        context = "\n".join(results["documents"][0])
    except Exception as e:
        context = f"Vectorstore unavailable: {e}"
    return {**state, "retrieved_context": context}


def get_rul_snapshot(state: AgentState) -> AgentState:
    try:
        path = os.path.join(DATA_PROCESSED_DIR, "features.parquet")
        df = pd.read_parquet(path)
        latest = df.sort_values("cycle").groupby("unit").last().reset_index()
        preds = predict_rul(latest)
        latest["predicted_rul"] = preds.clip(min=0)

        critical = latest[latest["predicted_rul"] <= RUL_CRITICAL_THRESHOLD]
        warning = latest[
            (latest["predicted_rul"] > RUL_CRITICAL_THRESHOLD) &
            (latest["predicted_rul"] <= RUL_WARNING_THRESHOLD)
        ]

        summary = {
        "total_units":          int(len(latest)),
        "critical":             int(len(critical)),
        "warning":              int(len(warning)),
        "healthy":              int(len(latest) - len(critical) - len(warning)),
        "critical_units":       [int(u) for u in critical["unit"].tolist()[:10]],
        "avg_fleet_rul":        round(float(latest["predicted_rul"].mean()), 1),
        "units_with_anomalies": int(df.groupby("unit")["is_anomaly"].any().sum()),
        "avg_anomaly_rate":     round(float(df["is_anomaly"].mean() * 100), 1)
        }
        rul_data = json.dumps(summary)
    except Exception as e:
        import traceback
        traceback.print_exc()
        rul_data = json.dumps({"error": str(e)})

    return {**state, "rul_data": rul_data}


def generate_response(state: AgentState) -> AgentState:
    """LLM generates response using retrieved context + RUL data."""
    llm = get_llm()
    prompt = f"""You are FailSight, an AI assistant for predictive maintenance and fleet reliability.

User query: {state['query']}

Fleet RUL snapshot (live):
{state['rul_data']}

Retrieved fleet context:
{state['retrieved_context']}
Be consistent - always refer to the same units from the live snapshots above.

Answer the query clearly and concisely. If units are critical or have anomalies, highlight them.
If the query asks for a failure report, generate a structured one with:
- Unit ID
- Current cycle
- Predicted RUL
- Risk level
- Recommended action
"""
    response = llm.invoke(prompt)
    return {**state, "response": response.content, "messages": [response.content]}


def should_generate_report(state: AgentState) -> str:
    """Route: if query mentions 'report' or 'critical', go deeper."""
    keywords = ["report", "critical", "failure", "breakdown", "urgent"]
    if any(k in state["query"].lower() for k in keywords):
        return "generate_report"
    return "respond"


def generate_failure_report(state: AgentState) -> AgentState:
    """Auto-generate a structured failure report for critical units."""
    llm = get_llm()
    rul_data = json.loads(state["rul_data"])

    prompt = f"""Generate a detailed failure report for a predictive maintenance system.

Fleet data:
{json.dumps(rul_data, indent=2)}

Context:
{state['retrieved_context']}

Format the report as:
## FailSight Failure Report

### Executive Summary
[brief summary]

### Critical Units
[list each critical unit with: Unit ID, Predicted RUL, Risk, Recommended Action]

### Fleet Health Overview
[overall health stats]

### Recommended Actions
[prioritized action list]
"""
    report = llm.invoke(prompt)
    return {**state, "report": report.content, "response": report.content, "messages": [report.content]}


# ── Graph ──────────────────────────────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_context)
    graph.add_node("rul_snapshot", get_rul_snapshot)
    graph.add_node("respond", generate_response)
    graph.add_node("generate_report", generate_failure_report)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rul_snapshot")
    graph.add_conditional_edges(
        "rul_snapshot",
        should_generate_report,
        {
            "generate_report": "generate_report",
            "respond": "respond"
        }
    )
    graph.add_edge("respond", END)
    graph.add_edge("generate_report", END)

    return graph.compile()


def run_agent(query: str) -> str:
    agent = build_agent()
    result = agent.invoke({
        "query": query,
        "retrieved_context": "",
        "rul_data": "",
        "report": "",
        "response": "",
        "messages": []
    })
    return result["response"]


if __name__ == "__main__":
    print(run_agent("Which units are critical right now?"))
    print(run_agent("Generate a failure report for this week"))