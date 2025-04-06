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

class MedicalMASUI:

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") 
    openai.api_key = api_key

    def __init__(self):
        if "query" not in st.session_state:
            st.session_state.query = ""
        if "generating" not in st.session_state:
            st.session_state.generating = False
        if "result" not in st.session_state:
            st.session_state.result = None
        if "query_logged" not in st.session_state:
            st.session_state.query_logged = False

    def generate_medical_aid(self, query):
        if not query.strip():
            st.warning("⚠️ Please enter a valid query before generating a solution.")
            return None
        crew = MedicalCrew(query=query)
        try:
            result = crew.run()
            return result
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

    def display_results(self, result):
        if not result:
            return

        tasks_output = getattr(result, 'tasks_output', [])
        if not isinstance(tasks_output, list):
            st.warning("⚠️ tasks_output is not valid.")
            tasks_output = []

        agent_roles = [
            ("Symptoms Analysis Agent", tasks_output[0] if len(tasks_output) > 0 else "No output provided by this agent."),
            ("Advisor Agent", tasks_output[1] if len(tasks_output) > 1 else "No output provided by this agent."),
            ("Verification Agent", tasks_output[2] if len(tasks_output) > 2 else "No output provided by this agent."),
            ("Estimated User Proficiency", tasks_output[3] if len(tasks_output) > 3 else "No output provided by this agent."),
        ]

        st.success("Your Medical Solution is Ready!")
        st.write("## **Agent Outputs and Thoughts**")

        master_agent_output = tasks_output[5] if len(tasks_output) > 5 else "No output provided by this agent."
        st.header("Master Agent")

        message = [{"role": "assistant", "content": """
            You need to take in the list of advice and return short search query to find an image based on the response.
            I want nothing else in your output other than the sentence.
        """}]

        pprint(vars(st.session_state.result))
        information = str(tasks_output[5])
        message.append({"role": "user", "content": information})
        chat_response = openai.chat.completions.create(model="gpt-4o-mini", messages=message)
        reply = chat_response.choices[0].message.content
        print(reply)

        imageURL = getImage(str(reply))
        if imageURL:
            imageURL = imageURL[0]
        else:
            st.error("No image is available for this query")

        if imageURL:
            st.markdown(
                f"""
                    <div style="position: relative; overflow: hidden; line-height: 1.6;">
                        <img src="{imageURL}" alt="Generated Image"
                            style="float: left; width: 300px; height: auto; margin-right: 20px; margin-bottom: 10px; border-radius: 10px;">
                        <p style="text-align: justify;">{str(master_agent_output).replace('.', '.<br>')}</p>
                    </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"<p>{str(master_agent_output).replace('.', '.<br>')}</p>", unsafe_allow_html=True)

        for role, agent_output in agent_roles:
            with st.expander(f"{role}"):
                formatted_output = str(agent_output).replace(".\n", ". ")
                st.markdown(f"<p>{formatted_output}</p>", unsafe_allow_html=True)

        st.markdown(
            """
            <div style='background-color: #1b4332; color: #ffffff; padding: 15px; border-radius: 5px; margin-top: 20px; margin-bottom: 20px; text-align: center;'>
            <strong>Emergency personnel have been notified. Stay calm and await assistance.</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Speak"):
            speech = str(tasks_output[5])
            tts = gTTS(speech)
            tts.save("output.mp3")
            st.audio("output.mp3")

    # ✅ NEW: Recognize speech input from the mic
    def recognize_speech_from_mic(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            st.info("Listening... Please speak your query.")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            st.info("Transcribing...")
            text = recognizer.recognize_google(audio)
            st.success(f"✅ You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError:
            st.error("Could not connect to the speech recognition service.")
        return ""

    def run(self):
        st.set_page_config(page_title="Medical MAS", page_icon="🩹")
        st.sidebar.title("Medical MAS Control Panel")
        st.sidebar.text("Use this panel to interact with the system.")

        if st.sidebar.button("Use Voice Input"):
            voice_text = self.recognize_speech_from_mic()
            if voice_text:
                st.session_state.query = voice_text

        st.sidebar.text_input(
            "Enter a medical query:", value=st.session_state.query, key="query"
        )

        if st.session_state.query.strip() == "":
            st.session_state.query_logged = False

        if st.sidebar.button("Generate Solution"):
            st.session_state.generating = True
            if not st.session_state.query_logged:
                with open("memory.txt", "a") as fileHandler:
                    fileHandler.write("Query: " + str(st.session_state.query) + "\n")
                st.session_state.query_logged = True

        if st.session_state.query.strip() == "":
            st.title("Welcome to the Medical MAS System")
            st.markdown(
                """
                **What is a Multi-Agent System (MAS)?**

                A Multi-Agent System (MAS) is a system where multiple agents (software entities) collaborate, communicate, and solve tasks together. Each agent has its own role and knowledge base, contributing to a shared goal. In this medical MAS, different agents analyze symptoms, provide medical advice, and verify data, helping to create a comprehensive solution for your medical query.
                """
            )

        elif st.session_state.generating:
            result = self.generate_medical_aid(st.session_state.query)
            st.session_state.result = result
            st.session_state.generating = False

        self.display_results(st.session_state.result)


def cleaMemory():
    open('memory.txt', 'w').close()
atexit.register(cleaMemory)

if __name__ == "__main__":
    ui = MedicalMASUI()
    ui.run()
