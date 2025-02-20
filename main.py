import os
from crewai import Crew, Process
from agents import MedicalAgents
from tasks import MedicalTasks
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class MedicalCrew:
    def __init__(self, query):
        self.query = query
        self.OpenAIGPT4 = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1500, # type: ignore
            openai_api_key=os.environ.get("OPENAI_API_KEY") # type: ignore
        )
    



    def run(self):
        # Initialize agents and tasks
        agents = MedicalAgents()
        tasks = MedicalTasks()
        masterAgent = agents.masterAgent()
        symptomAnalysisAgent = agents.symptomAnalysisAgent()
        advisorAgent = agents.advisorAgent()
        riskAssessmentAgent = agents.riskAssesmentAgent()
        verificationAgent = agents.verificationAgent()
        userProficiencyAgent = agents.userProficiencyAgent()



        # Define tasks
        classifySymptoms = tasks.classifySymptoms(
            agent=symptomAnalysisAgent,
            query=self.query,

        )

        recommendProtocol = tasks.recommendProtocol(
            agent=advisorAgent,
            context=[classifySymptoms],
            query= self.query
        )

        verifyRecommendation = tasks.verifyRecommendation(
            agent=verificationAgent,
            context=[recommendProtocol]
        )

        checkUserMedicalKnowledge = tasks.checkUserMedicalKnowledge(
            agent=userProficiencyAgent,
            query=self.query
        )

        escalateRisk = tasks.escelateRisk(
            agent=riskAssessmentAgent,
            context=[recommendProtocol]
        )

        userExplanation = tasks.userExplination(
            agent=masterAgent,
            context=[classifySymptoms, recommendProtocol, verifyRecommendation],
        )

        
        # Configure Crew
        crew = Crew(
            agents=[
                masterAgent,
                symptomAnalysisAgent,
                advisorAgent,
                riskAssessmentAgent,
                verificationAgent,
                userProficiencyAgent
            ],
            tasks=[
                classifySymptoms,
                recommendProtocol,
                verifyRecommendation,
                checkUserMedicalKnowledge,
                escalateRisk,
                userExplanation
            ],
            process=Process.sequential,
            # memory=True,
            verbose=True,
        )

        try:
            result = crew.kickoff()
            return result
        except Exception as e:
            print(f"Error during Crew execution: {e}")
            return {"error": str(e)}

