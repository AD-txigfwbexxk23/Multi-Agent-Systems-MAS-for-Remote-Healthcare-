import os
import openai
from dotenv import load_dotenv
import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
from qiskit.quantum_info import partial_trace, DensityMatrix
import numpy as np

import textstat

#Initialize AI 
load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
openai.api_key = api_key

#Initialize Classic Simulator -Not used in this code
simulator = AerSimulator()

#Initialize Variables
totalAngle = 0
totalSentimentAngle = 0
totalClarityAngle = 0
totalRelianceAngle = 0
totalTaskCompletionAngle = 0
totalAccuracyAngle = 0

previousRelianceAngle = 0

# Quantum Circuit Initialization - 0 for trust, 1 for sentiment, 2 for reliance, 3 for task completion, 4 for clarity, 5 for accuracy
qBits = QuantumCircuit(6)

#Apply Superposition
qBits.h(0)

def showCircuit():
    print(qBits.draw())

def readability_score(text):
    """Computes a single readability score in the range [-1,1] using a simple weighted average"""
    
    # Get readability metrics
    flesch = textstat.flesch_reading_ease(text)  # Higher = easier
    fk_grade = textstat.flesch_kincaid_grade(text)  # Lower = easier
    gunning_fog = textstat.gunning_fog(text)  # Lower = easier
    smog = textstat.smog_index(text)  # Lower = easier
    ari = textstat.automated_readability_index(text)  # Lower = easier
    dale_chall = textstat.dale_chall_readability_score(text)  # Lower = easier

    # Convert all metrics to the same scale (higher = easier)
    max_flesch = 100
    max_grade = 20  # Approximate max grade level
    max_dale_chall = 10

    normalized_scores = [
        flesch / max_flesch,  # Already 0-1
        1 - (fk_grade / max_grade),
        1 - (gunning_fog / max_grade),
        1 - (smog / max_grade),
        1 - (ari / max_grade),
        1 - (dale_chall / max_dale_chall),
    ]

    print(normalized_scores)

    # Ensure values are within [0,1]
    normalized_scores = [max(min(s, 1), 0) for s in normalized_scores]

    combined_score = np.mean(normalized_scores)

    # Scale to [-1,1]
    return 2 * combined_score - 1

def initialTrust(userInput):

    theta = (((userInput-5)*0.2)*math.pi/2)

    if userInput == 0 or userInput == 10:
        theta = (((userInput-5)*0.19)*math.pi/2)

    qBits.ry(theta, 0)

def riskAnalysis(userInput):
    global totalAngle
    
    risk_factor = (userInput / 10)
    theta = -risk_factor * (math.pi / 8)
    
    qBits.ry(theta, 0)

def priorKnowledgeAnalysis(userInput):
    global totalRelianceAngle
    
    knowledge_factor = (userInput / 10)  
    theta = -knowledge_factor * (math.pi / 8)  
    
    qBits.ry(theta, 2)

def sentimentAnalysis(userInput, counter):
    global totalAngle, totalSentimentAngle

    totalAngle = getCurrentAngle(qBits, 0)

    print(f"Total sentiment angle is: {totalSentimentAngle}")
    print(f"Total trust angle is: {totalAngle}")
    
    sentimentBackground=[{
        "role": "system", 
        "content": 
        """
        Your task is to analyze the sentence given to you and tell me its sentiment. Give it a score ranging from -1 to 1 depdening on how positive it is.
        (e.g. Anything greater then 0 is positve and anything below 0 is negative AND 0 is exactly neutral). 
        Only give me a number and nothing else. For example if I put I hate you that is very negative so the answer would be very close to -1.
        The closer to the 1 or -1 the better the sentiment.
        Factor in the exclamation marks, extra question marks for frustation, capitalizations for urgency and frustation etc. All of these could affect the sentiment. 
        If the user is just asking a simple question (for example, I am choking, what should I do?) that is near to neutral (0).
        IT IS IMPORTANT THAT YOU FACTOR IN THE CONTEXT WHILE DECIDING ON A SENTIMENT. MAKE SURE TO LOOK AT THE CONVERSATIONS ABOVE THE ONE BEING JUDGED IN ORDER TO GIVE A NUMBER.
        """
        }]
    
    sentimentBackground.append({"role": "user", "content": userInput})
    sentimentAnalysisAI= openai.chat.completions.create(model="gpt-4o-mini",messages=sentimentBackground)
    sentimentAnalysis= sentimentAnalysisAI.choices[0].message.content
    sentimentAnalysisVal= float(sentimentAnalysis)
    print(f"Sentiment Score: {sentimentAnalysisVal}")

    # Calculate the remaining angle for sentiment adjustment
    leftOverTheta = math.pi - totalSentimentAngle 

    if totalSentimentAngle < 0:
        leftOverTheta = -(-math.pi - totalSentimentAngle)
    
    theta = sentimentAnalysisVal * leftOverTheta /counter

    print(f"SentimentTheta: {theta}")

    qBits.ry(theta, 1)

    totalSentimentAngle += theta

    if sentimentAnalysisVal < 0:
        qBits.cry((0 - totalAngle)/counter, 1, 0)
    elif sentimentAnalysisVal > 0:
        qBits.cry((math.pi - totalAngle)/counter, 1, 0)
    
    totalAngle = getCurrentAngle(qBits, 0)

    print(f"Updated total trust angle: {totalAngle}")

def updateTaskCompletion(followUpQuestions, counter):
    global totalRelianceAngle, totalTaskCompletionAngle

    totalRelianceAngle = getCurrentAngle(qBits, 2)

    print(f"Total taskcompletion angle is: {totalTaskCompletionAngle}")
    print(f"Total reliance angle is: {totalRelianceAngle}")

    print(f"Number of Follow Up Questions: {followUpQuestions}")
    
    if followUpQuestions > 2:
        leftOverTheta = math.pi - totalTaskCompletionAngle

        decrement_factor = (followUpQuestions - 2) * 0.1
        theta = -decrement_factor * leftOverTheta / counter
        
        qBits.ry(theta, 3)

        qBits.cry((0 - totalRelianceAngle)/counter, 3, 2)

        totalTaskCompletionAngle += theta

    totalRelianceAngle = getCurrentAngle(qBits, 2)

    print(f"Updated total reliance angle: {totalRelianceAngle}")

def updateClarity(aiResponse, counter):
    global totalRelianceAngle, totalClarityAngle

    totalRelianceAngle = getCurrentAngle(qBits, 2)

    print(f"Total clarity angle is: {totalClarityAngle}")
    print(f"Total reliance angle is: {totalRelianceAngle}")
    
    clarityAnalysisVal = readability_score(str(aiResponse))

    print(f"Clarity Score: {clarityAnalysisVal}")

    # Calculate the remaining angle for sentiment adjustment
    leftOverTheta = math.pi - totalClarityAngle 

    if totalClarityAngle < 0:
        leftOverTheta = -(-math.pi - totalClarityAngle)
    
    theta = (clarityAnalysisVal * leftOverTheta)/counter

    print(f"ClarityTheta: {theta}")

    qBits.ry(theta, 4)

    totalClarityAngle += theta

    if clarityAnalysisVal < 0:
        qBits.cry((0 - totalRelianceAngle)/counter, 4, 2)
    elif clarityAnalysisVal > 0:
        qBits.cry((math.pi - totalRelianceAngle)/counter, 4, 2)

    totalRelianceAngle = getCurrentAngle(qBits, 2)

    print(f"Updated total reliance angle: {totalRelianceAngle}")

def updateAccuracy(aiResponse, counter):
    global totalRelianceAngle, totalAccuracyAngle

    totalRelianceAngle = getCurrentAngle(qBits, 2)

    print(f"Total accuracy angle is: {totalAccuracyAngle}")
    print(f"Total reliance angle is: {totalRelianceAngle}")
    
    accuracyBackground=[{
        "role": "system", 
        "content": 
        """
        Your task is to analyze the text given to you and tell me its accuracy (i.e. fact check). Give it a score ranging from -1 to 1 depending on how accurate it is.
        (e.g. Anything greater than 0 is going towards accuracy and anything below 0 is going towards inaccuracy). 

        Fact check from a number of sources online to gather the score, and make sure to give a number that is as correct as possible.
        You can get a number in between 0, 1 and -1 as well if some parts of the text is true and others are false. 
        (For example, if you have text of 2 sentences and one of them is completely false while other is completely correct, it should be around 0.5)

        Give a score of 0 for sentences that seem to be opinions or subjective as well as ones that are not fact checkable (i.e. response to Hello)

        Only give me a number and nothing else. 
        """
    }]
    
    accuracyBackground.append({"role": "user", "content": str(aiResponse)})
    accuracyAnalysisAI= openai.chat.completions.create(model="gpt-4o-mini",messages=accuracyBackground)
    accuracyAnalysis= accuracyAnalysisAI.choices[0].message.content
    accuracyAnalysisVal= float(accuracyAnalysis)

    print(f"Accuracy Analysis Score: {accuracyAnalysisVal}")

    # Calculate the remaining angle for sentiment adjustment
    leftOverTheta = math.pi - totalAccuracyAngle 

    if totalAccuracyAngle < 0:
        leftOverTheta = -(-math.pi - totalAccuracyAngle)
    
    theta = accuracyAnalysisVal * leftOverTheta /counter

    print(f"Accuracy Analysis Theta: {theta}")

    qBits.ry(theta, 5)

    totalAccuracyAngle += theta

    if accuracyAnalysisVal < 0:
        qBits.cry((0 - totalRelianceAngle)/counter, 5, 2)
    elif accuracyAnalysisVal > 0:
        qBits.cry((math.pi - totalRelianceAngle)/counter, 5, 2)

    totalRelianceAngle = getCurrentAngle(qBits, 2)

def updateReliance(aiResponse, followUpQuestions, counter):
    updateClarity(str(aiResponse), counter)
    updateAccuracy(str(aiResponse), counter)
    updateTaskCompletion(followUpQuestions, counter)

    #Update Trust

    global totalAngle, totalRelianceAngle, previousRelianceAngle

    totalRelianceAngle = getCurrentAngle(qBits, 2)
    totalAngle = getCurrentAngle(qBits, 0)

    print(f"Total reliance angle is: {totalRelianceAngle}")
    print(f"Total trust angle is: {totalAngle}")

    if (totalRelianceAngle - previousRelianceAngle) < 0:
        qBits.cry((0 - totalAngle)/counter, 2, 0)
    elif (totalRelianceAngle - previousRelianceAngle) > 0:
        qBits.cry((math.pi - totalAngle)/counter, 2, 0)

    previousRelianceAngle = totalRelianceAngle

    totalAngle = getCurrentAngle(qBits, 0)

    print(f"Updated total trust angle: {totalAngle}")

def updateTrust(userInput, aiResponse, counter, followUpQuestions):
    sentimentAnalysis(userInput, counter)
    updateReliance(aiResponse, followUpQuestions, counter)

    showCircuit()

def getSingleQubitProbabilities():
    # Obtain the statevector from the quantum circuit
    state = Statevector.from_instruction(qBits)
    
    # Compute the reduced density matrix for qubit 0 by tracing out qubit 1
    reduced_dm = partial_trace(state, [1])
    reduced_dm = partial_trace(reduced_dm, [1])
    reduced_dm = partial_trace(reduced_dm, [1])
    reduced_dm = partial_trace(reduced_dm, [1])
    reduced_dm = partial_trace(reduced_dm, [1])

    # Convert the reduced state to a DensityMatrix
    dm = DensityMatrix(reduced_dm)
    
    # Extract the diagonal elements to get probabilities
    prob_0 = np.real(dm.data[0, 0])  # Element corresponding to |0><0|
    prob_1 = np.real(dm.data[1, 1])  # Element corresponding to |1><1|
    
    print(f"Probability of qubit 0 being 0: {prob_0}")
    print(f"Probability of qubit 0 being 1: {prob_1}")
    
    return prob_1

def showQubit():
    
    state = Statevector.from_instruction(qBits)

    # Step 5: Visualize the state on the Bloch sphere
    plot_bloch_multivector(state.data)

    # Step 6: Display the plot
    plt.show()

def getCurrentAngle(qBits, qubit_index=0):
    # Obtain the state vector from the quantum circuit
    state = Statevector.from_instruction(qBits)
    
    # Compute the reduced density matrix for the qubit of interest
    reduced_dm = partial_trace(state, [i for i in range(qBits.num_qubits) if i != qubit_index])
    
    # Convert to a DensityMatrix object
    dm = DensityMatrix(reduced_dm)
    
    # Access the elements of the density matrix
    rho = dm.data  # Get the NumPy array representing the density matrix
    
    # Extract real and imaginary parts of the elements to compute Bloch vector
    rho00 = np.real(rho[0, 0])
    rho01 = rho[0, 1]
    rho10 = rho[1, 0]
    rho11 = np.real(rho[1, 1])
    
    # Compute Bloch vector components
    x = 2 * np.real(rho01)
    y = 2 * np.imag(rho01)
    z = rho00 - rho11
    
    # Calculate the angle theta from the Z-axis projection
    theta = np.arccos(z)  # z is the Z component of the Bloch vector
    
    # Return the angle in radians
    return theta