import streamlit as st
import openai 
from main import MedicalCrew
from imageGeneration import getImage
from dotenv import load_dotenv
import os
from pprint import pprint
from gtts import gTTS
import atexit
import speech_recognition as sr
import trustModel as trust
import math
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import json

class MedicalMASUI:

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") 
    openai.api_key = api_key

    def __init__(self):
        if "query" not in st.session_state:
            # Clear CSV at the beginning of a new session
            with open("trust_scores.csv", "w") as f:
                f.write("TrustScoreRatio\n")
        if "generating" not in st.session_state:
            st.session_state.generating = False
        if "result" not in st.session_state:
            st.session_state.result = None
        if "query_logged" not in st.session_state:
            st.session_state.query_logged = False

        # Quantum trust model specific session states
        if "trustScoresRatio" not in st.session_state:
            st.session_state.trustScoresRatio = []
        if "button_pressed" not in st.session_state:
            st.session_state.button_pressed = False
        if "counter" not in st.session_state:
            st.session_state.counter = 2
        if "follow_up_questions" not in st.session_state:
            st.session_state.follow_up_questions = 0
        if "task_completed" not in st.session_state:
            st.session_state.task_completed = False

    def save_trust_scores_to_csv(self):
        df = pd.DataFrame(st.session_state.trustScoresRatio, columns=["TrustScoreRatio"])
        df.to_csv("trust_scores.csv", index=False)

    def show_trust_plot(self):
        if len(st.session_state.trustScoresRatio) > 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(len(st.session_state.trustScoresRatio)), st.session_state.trustScoresRatio, label="Trust Score (%)")
            ax.set_title("Trust Evolution")
            ax.set_xlabel("Interaction #")
            ax.set_ylabel("Trust %")
            ax.legend()
            st.pyplot(fig)

    def update_quantum_trust(self, user_input, ai_response):
        trust.updateTrust(user_input, ai_response, st.session_state.counter, st.session_state.follow_up_questions)
        counts = trust.getSingleQubitProbabilities()
        ratio = counts * 100
        st.session_state.trustScoresRatio.append(ratio)
        self.save_trust_scores_to_csv()
        st.session_state.counter += 0.5

    def generate_medical_aid(self, query):
        crew = MedicalCrew(query=query)
        result = crew.run()
        return result

    def display_results(self, result):
        if not result:
            return

        tasks_output = getattr(result, 'tasks_output', [])
        agent_outputs = [
            ("Symptoms Analysis Agent", tasks_output[0] if len(tasks_output) > 0 else "No output."),
            ("Advisor Agent", tasks_output[1] if len(tasks_output) > 1 else "No output."),
            ("Verification Agent", tasks_output[2] if len(tasks_output) > 2 else "No output."),
            ("Estimated User Proficiency", tasks_output[3] if len(tasks_output) > 3 else "No output.")
        ]
        master_agent_output = tasks_output[5] if len(tasks_output) > 5 else "No master output."

        # Update quantum trust model
        self.update_quantum_trust(st.session_state.query, master_agent_output)

        # Conditional trust adjustment
        trust_level = st.session_state.trustScoresRatio[-1] if st.session_state.trustScoresRatio else 50
        st.success(f"✅ Current Trust Score: {trust_level:.2f}%")

        if trust_level < 40:
            st.warning("⚠️ Trust is relatively low. Proceed carefully and consider double-checking this advice.")


        # Generate image
        message = [{"role": "assistant", "content": """
            Turn the following advice into a short search query for an image. 
            Output only the search phrase.
        """}]
        message.append({"role": "user", "content": str(master_agent_output)})
        reply = openai.chat.completions.create(model="gpt-4o-mini", messages=message).choices[0].message.content
        imageURL = getImage(reply)
        st.header("Master Agent Summary")

        if imageURL:
            imageURL = imageURL[0]
            col1, col2 = st.columns([5, 6])
            with col1:
                st.image(imageURL, caption="Contextual Image", use_container_width=True)
            with col2:
                st.markdown(f"<p>{str(master_agent_output).replace('.', '.<br>')}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p>{str(master_agent_output).replace('.', '.<br>')}</p>", unsafe_allow_html=True)


        # Display agent outputs
        for role, output in agent_outputs:
            with st.expander(role):
                st.markdown(f"<p>{str(output).replace('.', '.<br>')}</p>", unsafe_allow_html=True)

        # Speak
        if st.button("Speak"):
            tts = gTTS(master_agent_output)
            tts.save("output.mp3")
            st.audio("output.mp3")

    def recognize_speech_from_mic(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            st.info("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            st.info("Transcribing...")
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand.")
        except sr.RequestError:
            st.error("Speech service error.")
        return ""

    def run(self):
        st.set_page_config(page_title="Quantum MAS", page_icon="🩹")

        if not st.session_state.button_pressed:
            st.title("Initialize Trust Model")
            initialTrust = st.slider("Trust in AI (0-10)", 0, 10, 5)
            situationRisk = st.slider("Situation Risk (0-10)", 0, 10, 5)
            priorKnowledge = st.slider("Your Knowledge (0-10)", 0, 10, 5)
            if st.button("Confirm"):
                trust.initialTrust(initialTrust)
                trust.priorKnowledgeAnalysis(priorKnowledge)
                trust.riskAnalysis(situationRisk)
                ratio = trust.getSingleQubitProbabilities() * 100
                st.session_state.trustScoresRatio.append(ratio)
                st.session_state.button_pressed = True
                st.rerun()
            return

        st.sidebar.title("MAS Control")
        if st.sidebar.button("Use Voice Input"):
            text = self.recognize_speech_from_mic()
            if text:
                st.session_state.query = text

        st.sidebar.text_input("Enter query:", key="query")
        if st.sidebar.button("Generate Solution"):
            st.session_state.generating = True

        if st.session_state.generating:
            result = self.generate_medical_aid(st.session_state.query)
            st.session_state.result = result
            st.session_state.generating = False

        self.display_results(st.session_state.result)

def clearMemory():
    open("memory.txt", "w").close()
atexit.register(clearMemory)

if __name__ == "__main__":
    ui = MedicalMASUI()
    ui.run()
