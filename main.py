from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Airline Satisfaction Prediction API",
    description="Predicts airline passenger satisfaction using a Random Forest model.",
    version="1.0.0"
)

# Load trained ML pipeline
model = joblib.load("model.pkl")


class PassengerData(BaseModel):
    Gender: str
    Age: int
    Customer_Type: str
    Type_of_Travel: str
    Class: str
    Flight_Distance: int
    Departure_Delay: float
    Arrival_Delay: float
    Departure_and_Arrival_Time_Convenience: int
    Ease_of_Online_Booking: int
    Check_in_Service: int
    Online_Boarding: int
    On_board_Service: int
    Seat_Comfort: int
    Leg_Room_Service: int
    Cleanliness: int
    Food_and_Drink: int
    In_flight_Service: int
    In_flight_Wifi_Service: int
    In_flight_Entertainment: int
    Baggage_Handling: int


@app.get("/")
def home():
    return {
        "message": "Airline Satisfaction Prediction API is running"
    }


@app.post("/predict")
def predict(data: PassengerData):

    input_data = pd.DataFrame([{
        "Gender": data.Gender,
        "Age": data.Age,
        "Customer Type": data.Customer_Type,
        "Type of Travel": data.Type_of_Travel,
        "Class": data.Class,
        "Flight Distance": data.Flight_Distance,
        "Departure Delay": data.Departure_Delay,
        "Arrival Delay": data.Arrival_Delay,
        "Departure and Arrival Time Convenience":
            data.Departure_and_Arrival_Time_Convenience,
        "Ease of Online Booking": data.Ease_of_Online_Booking,
        "Check-in Service": data.Check_in_Service,
        "Online Boarding": data.Online_Boarding,
        "On-board Service": data.On_board_Service,
        "Seat Comfort": data.Seat_Comfort,
        "Leg Room Service": data.Leg_Room_Service,
        "Cleanliness": data.Cleanliness,
        "Food and Drink": data.Food_and_Drink,
        "In-flight Service": data.In_flight_Service,
        "In-flight Wifi Service": data.In_flight_Wifi_Service,
        "In-flight Entertainment": data.In_flight_Entertainment,
        "Baggage Handling": data.Baggage_Handling
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        result = "Satisfied"
    else:
        result = "Neutral or Dissatisfied"

    return {
    "prediction": int(prediction),
    "result": result,
    "probability_neutral_or_dissatisfied": float(probability[0]),
    "probability_satisfied": float(probability[1])
    }