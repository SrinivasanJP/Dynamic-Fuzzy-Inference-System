from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import skfuzzy as fuzz
import numpy as np


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define the model for the request data
class HealthData(BaseModel):
    name: str
    temperature: float
    heart_rate: int
    blood_pressure: int
    respiratory_rate: int
    oxygen_saturation: float
    blood_sugar: float

# Set up the fuzzy inference system with additional parameters
def create_fis():
    fis = {}

    # Define fuzzy membership functions for Temperature
    temperature_range = np.arange(35, 42.1, 0.1)
    fis['temperature_normal'] = fuzz.trapmf(temperature_range, [35, 35, 36.5, 37.5])
    fis['temperature_fever'] = fuzz.trapmf(temperature_range, [36.5, 37.5, 39, 40])
    fis['temperature_high_fever'] = fuzz.trapmf(temperature_range, [39, 40, 42, 42])

    # Define fuzzy membership functions for Heart Rate
    heart_rate_range = np.arange(40, 181, 1)
    fis['heart_rate_normal'] = fuzz.trapmf(heart_rate_range, [40, 40, 60, 80])
    fis['heart_rate_elevated'] = fuzz.trapmf(heart_rate_range, [70, 100, 120, 150])
    fis['heart_rate_tachycardia'] = fuzz.trapmf(heart_rate_range, [120, 140, 180, 180])

    # Define fuzzy membership functions for Blood Pressure
    blood_pressure_range = np.arange(90, 181, 1)
    fis['bp_normal'] = fuzz.trapmf(blood_pressure_range, [90, 90, 110, 120])
    fis['bp_elevated'] = fuzz.trapmf(blood_pressure_range, [110, 130, 140, 150])
    fis['bp_hypertension'] = fuzz.trapmf(blood_pressure_range, [140, 160, 180, 180])

    # Define fuzzy membership functions for Respiratory Rate
    respiratory_rate_range = np.arange(12, 40, 1)
    fis['respiratory_rate_normal'] = fuzz.trapmf(respiratory_rate_range, [12, 12, 16, 20])
    fis['respiratory_rate_elevated'] = fuzz.trapmf(respiratory_rate_range, [18, 22, 28, 35])
    fis['respiratory_rate_high'] = fuzz.trapmf(respiratory_rate_range, [30, 35, 40, 40])

    # Define fuzzy membership functions for Oxygen Saturation
    oxygen_saturation_range = np.arange(85, 101, 1)
    fis['oxygen_normal'] = fuzz.trapmf(oxygen_saturation_range, [95, 95, 97, 100])
    fis['oxygen_low'] = fuzz.trapmf(oxygen_saturation_range, [85, 88, 90, 94])
    fis['oxygen_critical'] = fuzz.trapmf(oxygen_saturation_range, [85, 85, 90, 92])

    # Define fuzzy membership functions for Blood Sugar
    blood_sugar_range = np.arange(70, 300, 1)
    fis['sugar_normal'] = fuzz.trapmf(blood_sugar_range, [70, 70, 90, 110])
    fis['sugar_elevated'] = fuzz.trapmf(blood_sugar_range, [100, 130, 150, 180])
    fis['sugar_high'] = fuzz.trapmf(blood_sugar_range, [170, 200, 300, 300])

    # Define fuzzy membership functions for Diagnosis
    diagnosis_range = np.arange(0, 101, 1)
    fis['healthy'] = fuzz.trimf(diagnosis_range, [0, 0, 30])
    fis['feverish'] = fuzz.trimf(diagnosis_range, [20, 50, 80])
    fis['critical'] = fuzz.trimf(diagnosis_range, [70, 100, 100])

    return fis

# Instantiate the fuzzy inference system
fis = create_fis()

# Fuzzy rule evaluation function with additional parameters
# Fuzzy rule evaluation function with rule-based conditions
def evaluate_fis(data: HealthData, fis):
    # Map inputs to fuzzy values
    temperature_normal = fuzz.interp_membership(np.arange(35, 42.1, 0.1), fis['temperature_normal'], data.temperature)
    temperature_fever = fuzz.interp_membership(np.arange(35, 42.1, 0.1), fis['temperature_fever'], data.temperature)
    heart_rate_normal = fuzz.interp_membership(np.arange(40, 181, 1), fis['heart_rate_normal'], data.heart_rate)
    heart_rate_tachycardia = fuzz.interp_membership(np.arange(40, 181, 1), fis['heart_rate_tachycardia'], data.heart_rate)
    bp_normal = fuzz.interp_membership(np.arange(90, 181, 1), fis['bp_normal'], data.blood_pressure)
    bp_hypertension = fuzz.interp_membership(np.arange(90, 181, 1), fis['bp_hypertension'], data.blood_pressure)
    heart_rate_elevated = fuzz.interp_membership(np.arange(40, 181, 1), fis['heart_rate_elevated'], data.heart_rate)

    # Initialize diagnosis level
    diagnosis_level = 0

    # Apply rules based on the user's name
    if data.name == 'Srini':
        # Rules for Srini
        if temperature_normal and heart_rate_normal:
            diagnosis = "Healthy"
            diagnosis_level = 20
        elif temperature_fever:
            diagnosis = "Feverish"
            diagnosis_level = 50
        elif heart_rate_tachycardia:
            diagnosis = "Critical"
            diagnosis_level = 90
        else:
            diagnosis = "Uncertain"

    elif data.name == 'Gokul':
        # Rules for Gokul
        if temperature_normal:
            diagnosis = "Healthy"
            diagnosis_level = 20
        elif bp_hypertension:
            diagnosis = "Critical"
            diagnosis_level = 90
        elif heart_rate_elevated:
            diagnosis = "Feverish"
            diagnosis_level = 50
        else:
            diagnosis = "Uncertain"

    elif data.name == 'Hemavershini':
        # Rules for Hemavershini
        if temperature_normal and heart_rate_normal:
            diagnosis = "Healthy"
            diagnosis_level = 20
        elif temperature_fever:
            diagnosis = "Feverish"
            diagnosis_level = 50
        elif heart_rate_tachycardia:
            diagnosis = "Critical"
            diagnosis_level = 90
        else:
            diagnosis = "Uncertain"

    else:
        # Default general diagnosis if no user-specific rules are met
        diagnosis = "General"
        diagnosis_level = max(temperature_normal, heart_rate_normal, bp_normal)

    return diagnosis, diagnosis_level

# Endpoint to get diagnosis with rule-based evaluation
@app.post("/get_diagnosis/")
async def get_diagnosis(data: HealthData):
    # Validate inputs
    if not (35 <= data.temperature <= 42):
        raise HTTPException(status_code=400, detail="Temperature must be between 35 and 42 Celsius.")
    if not (40 <= data.heart_rate <= 180):
        raise HTTPException(status_code=400, detail="Heart rate must be between 40 and 180 bpm.")
    if not (90 <= data.blood_pressure <= 180):
        raise HTTPException(status_code=400, detail="Blood pressure must be between 90 and 180 mmHg.")
    if not (12 <= data.respiratory_rate <= 40):
        raise HTTPException(status_code=400, detail="Respiratory rate must be between 12 and 40 breaths per minute.")
    if not (85 <= data.oxygen_saturation <= 100):
        raise HTTPException(status_code=400, detail="Oxygen saturation must be between 85 and 100 percent.")
    if not (70 <= data.blood_sugar <= 300):
        raise HTTPException(status_code=400, detail="Blood sugar must be between 70 and 300 mg/dL.")

    # Evaluate FIS and get diagnosis level and advice
    diagnosis, diagnosis_level = evaluate_fis(data, fis)

    # Interpret diagnosis with advice
    if diagnosis == "Healthy":
        advice = "Keep up the good lifestyle! Regular check-ups and a balanced diet are recommended."
    elif diagnosis == "Feverish":
        advice = "Rest, stay hydrated, and monitor symptoms. Consult a doctor if symptoms persist."
    elif diagnosis == "Critical":
        advice = "Seek medical attention immediately for evaluation and possible treatment."
    else:
        advice = "Consult a healthcare professional for further evaluation."

    # Return the response
    return {
        "diagnosis": diagnosis,
        "advice": advice,
        "diagnosis_level": diagnosis_level
    }


# Run the server with:
# uvicorn server:app --reload
