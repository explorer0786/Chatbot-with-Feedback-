import streamlit as st
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command, interrupt
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid
import os

# ---------------------------
# ENVIRONMENT SETUP
# ---------------------------
os.environ["GROQ_API_KEY"] = "api_key"  # Replace with your Groq key
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# ---------------------------
# STATE DEFINITION
# ---------------------------
class State(TypedDict):
    query: str
    answer: Annotated[List[AIMessage], add_messages]
    human_feedback: Annotated[List[str], add_messages]

# ---------------------------
# MODEL NODE
# ---------------------------
def model(state: State):
    query = state["query"]
    feedback = state["human_feedback"][-1] if state["human_feedback"] else "No feedback yet"

    prompt = f"""
    Question/Request: {query}
    Human Feedback: {feedback}
    
    Based on the query and feedback, generate a refined response.
    """

    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ])
    result = response.content

    return {
        "answer": state.get("answer", []) + [AIMessage(content=result)],
        "human_feedback": state.get("human_feedback", [])
    }

# ---------------------------
# HUMAN FEEDBACK NODE
# ---------------------------
def human_node(state: State):
    answer = state["answer"][-1].content
    user_feedback = interrupt({
        "generated_answer": answer,
        "message": "Provide feedback or type 'done' to finish:"
    })

    if user_feedback.strip().lower() == "done":
        return Command(update={"human_feedback": state["human_feedback"] + ["Finalized"]}, goto="end_node")

    return Command(update={"human_feedback": state["human_feedback"] + [user_feedback]}, goto="model")

# ---------------------------
# END NODE
# ---------------------------
def end_node(state: State):
    return {"answer": state["answer"], "human_feedback": state["human_feedback"]}

# ---------------------------
# BUILD LANGGRAPH
# ---------------------------
graph = StateGraph(State)
graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("end_node", end_node)

graph.set_entry_point("model")
graph.add_edge(START, "model")
graph.add_edge("model", "human_node")
graph.add_edge("human_node", "model")  # Feedback leads back to model for refinement
graph.add_edge("human_node", "end_node")  # Finalization goes to end node
graph.set_finish_point("end_node")

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="centered")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "state" not in st.session_state:
    st.session_state.state = {
        "query": "",
        "answer": [],
        "human_feedback": []
    }
if "awaiting_feedback" not in st.session_state:
    st.session_state.awaiting_feedback = False

st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fb;
            color: #333;
        }
        .block-container {
            padding: 2rem;
            border-radius: 12px;
            background: linear-gradient(to right, #e0f7fa, #fce4ec);
        }
        .stTextInput>div>div>input {
            background-color: #fff !important;
            color: black !important;
        }
        .stButton>button {
            background-color: #009688 !important;
            color: white !important;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 Chatbot with Feedback")
st.markdown("Ask anything and refine with human feedback!")

# ---------------------------
# INPUT SECTION
# ---------------------------
query_input = st.text_input("Your Question:", key="user_query")
submit_clicked = st.button("💬 Submit", key="submit")

if submit_clicked and query_input.strip():
    st.session_state.state = {
        "query": query_input,
        "answer": [],
        "human_feedback": []
    }
    thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
    for chunk in app.stream(st.session_state.state, config=thread_config):
        for node_id, val in chunk.items():
            if node_id == "__interrupt__":
                st.session_state.awaiting_feedback = True
            elif isinstance(val, dict):
                st.session_state.state.update(val)
    st.experimental_rerun()

# ---------------------------
# DISPLAY GENERATED ANSWER
# ---------------------------
if st.session_state.state.get("answer"):
    latest_answer = st.session_state.state["answer"][-1].content
    st.markdown("### 🤖 Assistant's Answer:")
    st.info(latest_answer)

# ---------------------------
# FEEDBACK SECTION
# ---------------------------
if st.session_state.awaiting_feedback:
    feedback = st.text_input("✍️ Your Feedback (type 'done' to finalize)", key="user_feedback")
    if st.button("✅ Send Feedback", key="feedback"):
        updated_state = {
            "query": st.session_state.state["query"],
            "answer": st.session_state.state["answer"],
            "human_feedback": st.session_state.state["human_feedback"] + [feedback]
        }
        thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

        for chunk in app.stream(updated_state, config=thread_config):
            for node_id, val in chunk.items():
                if node_id == "__interrupt__":
                    st.session_state.awaiting_feedback = True
                elif isinstance(val, dict):
                    st.session_state.state.update(val)

        if feedback.strip().lower() == "done":
            st.session_state.awaiting_feedback = False
        st.experimental_rerun()

# ---------------------------
# FINAL OUTPUT
# ---------------------------
if st.session_state.state.get("human_feedback") and \
   st.session_state.state["human_feedback"][-1] == "Finalized":

    st.success("✅ Conversation Finalized")
    st.markdown("### 🧠 Final Answer:")
    st.success(st.session_state.state["answer"][-1].content)

    st.markdown("### 📋 Feedback Given:")
    for fb in st.session_state.state["human_feedback"][:-1]:
        st.write(f"- {fb}")
