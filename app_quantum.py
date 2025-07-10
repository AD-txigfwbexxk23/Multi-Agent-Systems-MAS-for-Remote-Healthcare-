import streamlit as st
import openai
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import trustModel as trust
import time  # Import time module
import pandas as pd

# Initialization
load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
openai.api_key = api_key

def save_trust_scores_to_csv(trust_scores, filename="trust_scores.csv"):  # NEW FUNCTION
    df = pd.DataFrame(trust_scores, columns=["TrustScoreRatio"])
    df.to_csv(filename, index=False)

# Store variables in session state to persist across reruns
if "counter" not in st.session_state:
    st.session_state.counter = 2

if "follow_up_questions" not in st.session_state:
    st.session_state.follow_up_questions = 0

if "task_completed" not in st.session_state:
    st.session_state.task_completed = False

# Streamlit App
st.set_page_config(page_title="Quantum-Enhanced AI Chatbot", layout="wide")
st.title("Quantum-Enhanced AI Chatbot")
st.write("Interact with the AI and watch the trust levels evolve in real-time!")

# Persistent State for Trust Scores and Timer
if "trustScoresRatio" not in st.session_state:
    st.session_state.trustScoresRatio = []

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": """
                                  You are a kind and helpful assistant. You should always try to give the quickest and best response possible to get participants out of the situation and tell them steps on how they could solve their problem."""}]

if "button_pressed" not in st.session_state:
    st.session_state.button_pressed = False

if not st.session_state.button_pressed:

    # User Input for Initial Trust
    initialTrust = st.slider("From 0-10, how much do you trust this AI system?", 0, 10, 5)
    situationRisk = st.slider("From 0-10, what is the risk involved in the situation?", 0, 10, 5)
    priorKnowledge = st.slider("From 0-10, how much prior knowledge do you have about the topic?", 0, 10, 5)

    if st.button("Confirm"):
        st.session_state.button_pressed = True  

        trust.initialTrust(initialTrust)
        trust.priorKnowledgeAnalysis(priorKnowledge)
        trust.riskAnalysis(situationRisk)
        counts = trust.getSingleQubitProbabilities()
        ratio = counts * 100
        st.session_state.trustScoresRatio.append(ratio)

        st.rerun()

else:

    # Chat Input and Output
    user_input = st.text_input("Your Message:")

    if st.button("Send"):
        if user_input:
            # Add User Message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Get AI Response
            chat_response = openai.chat.completions.create(model="gpt-4o-mini", messages=st.session_state.messages)
            reply = chat_response.choices[0].message.content
            st.write(f"AI: {reply}")
            st.session_state.messages.append({"role": "assistant", "content": reply})

            trust.updateTrust(user_input, reply, st.session_state.counter, st.session_state.follow_up_questions)
            counts = trust.getSingleQubitProbabilities()
            ratio = counts * 100
            st.session_state.trustScoresRatio.append(ratio)

            save_trust_scores_to_csv(st.session_state.trustScoresRatio)

            st.session_state.counter += 0.5  # Update counter in session state

    # Plot Trust Scores in Real-Time
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(range(len(st.session_state.trustScoresRatio)), st.session_state.trustScoresRatio, label="Trust Score Ratio")
    # ax.set_xticks(range(len(st.session_state.trustScoresRatio)))
    # ax.set_yticks(list(range(0, 101, 10)))
    # ax.set_title("Trust Level Evolution")
    # ax.set_xlabel("Interaction Count")
    # ax.set_ylabel("Trust Score")
    # ax.legend()
    # buf = BytesIO()
    # plt.savefig(buf, format="png")
    # st.image(buf)

    # Sidebar Layout
    with st.sidebar:
        st.header("Options")
        if st.button("Task Completed"):
            st.success("Task marked as completed!")
            st.session_state.task_completed = True  # Update in session state
        if st.button("New Chat"):
            st.session_state.button_pressed = False  
            st.rerun()

    if not st.session_state.task_completed:
        st.session_state.follow_up_questions += 1  # Update in session state
    else:
        st.session_state.follow_up_questions = 0  # Reset only if task is completed