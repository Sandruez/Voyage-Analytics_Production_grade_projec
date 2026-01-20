
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from voyage_analytics.constants import APP_HOST, APP_PORT
from voyage_analytics.pipline.prediction_pipeline import USvisaData, USvisaClassifier
from voyage_analytics.pipline.training_pipeline import TrainPipeline
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from typing import Optional
import pandas as pd
from datetime import datetime

# --- CONFIGURATION & SETUP ---


app = FastAPI()

# Mount static files (ensure you have a 'static' folder or create one)
# app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MOCK PREDICTION CLASSES (Replace with your actual voyage_analytics imports) ---

class FlightData:
    def __init__(self, from_loc, to_loc, flight_type, time, distance, agency, date):
        self.from_loc = from_loc
        self.to_loc = to_loc
        self.flight_type = flight_type
        self.time = time
        self.distance = distance
        self.agency = agency
        # Convert date to day name (e.g., "Monday")
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            self.day = dt.strftime("%A")
        except:
            self.day = "Monday"

    def get_data_as_dataframe(self):
        # Order matches Prompt Requirement #7
        data = {
            "from": [self.from_loc],
            "to": [self.to_loc],
            "flightType": [self.flight_type],
            "time": [self.time],
            "distance": [self.distance],
            "agency": [self.agency],
            "day": [self.day]
        }
        return pd.DataFrame(data)
    
class GenderData:
    def __init__(self, name, context):
        self.name = name
        self.context = context

class HotelData:
    def __init__(self, hotel_name, days, price_pref, pop_pref):
        self.hotel_name = hotel_name
        self.days = days
        self.price_pref = price_pref
        self.pop_pref = pop_pref

# --- REQUEST PARSING CLASS ---

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        # Flight Attributes
        self.travel_code: Optional[int] = None
        self.from_loc: Optional[str] = None
        self.to_loc: Optional[str] = None
        self.flight_type: Optional[str] = None
        self.price: Optional[float] = None
        self.time: Optional[float] = None
        self.distance: Optional[float] = None
        self.agency: Optional[str] = None
        self.date: Optional[str] = None
        # Gender Attributes
        self.full_name: Optional[str] = None
        self.context: Optional[str] = None
        # Hotel Attributes
        self.hotel_name: Optional[str] = None
        self.days: Optional[int] = None
        self.price_pref: Optional[str] = None
        self.pop_pref: Optional[str] = None

    async def get_flight_data(self):
        form = await self.request.form()
        self.travel_code = int(form.get("travelCode", 0))
        self.from_loc = form.get("from", "Unknown")
        self.to_loc = form.get("to", "Unknown")
        self.flight_type = form.get("flightType", "Unknown")
        # target variable
        self.price = float(form.get("price", 0.0))
        self.time = float(form.get("time", 0.0))
        self.distance = float(form.get("distance", 0.0))
        self.agency = form.get("agency", "Unknown")
        self.date = form.get("date", datetime.now().strftime("%Y-%m-%d"))

    async def get_gender_data(self):
        form = await self.request.form()
        self.full_name = form.get("fullName", "").strip()
        self.context = form.get("context", "Global")

    async def get_hotel_data(self):
        form = await self.request.form()
        self.hotel_name = form.get("hotelSelect", "").split(',')[0] # Take only name if value is complex
        self.days = int(form.get("stayDays", 1))
        self.price_pref = form.get("pricePref", "Any")
        self.pop_pref = form.get("popPref", "Any")

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("usvisa.html", {"request": request})

@app.post("/predict_flight")
async def predict_flight(request: Request):
    try:
        form = DataForm(request)
        await form.get_flight_data()
        
        # 1. Prepare Data
        flight_data = FlightData(
            from_loc=form.from_loc,
            to_loc=form.to_loc,
            flight_type=form.flight_type,
            time=form.time,
            distance=form.distance,
            agency=form.agency,
            date=form.date
        )
        
        # 2. Get DataFrame (This matches the order requested in Prompt #7)
        df = flight_data.get_data_as_dataframe()
        
        # 3. Simulate Prediction (Replace with actual model call)
        # model = FlightModelLoader()
        # prediction = model.predict(df)
        prediction_result = f"${form.price * 1.12:.2f}" # Dummy logic

        response_payload = {
            "task": "flight_price_prediction",
            "input_received": df.to_dict(orient="records")[0],
            "prediction": prediction_result
        }
        
        return templates.TemplateResponse(
            "usvisa.html", 
            {"request": request, "result_flight": response_payload, "active_tab": "flight"}
        )
    except Exception as e:
        return templates.TemplateResponse("usvisa.html", {"request": request, "error": str(e)})

@app.post("/predict_gender")
async def predict_gender(request: Request):
    try:
        form = DataForm(request)
        await form.get_gender_data()
        
        # Logic
        gender = "Female" if form.full_name.lower().endswith(('a', 'e', 'i')) else "Male" # Mock Logic
        
        response_payload = {
            "task": "gender_classification",
            "name": form.full_name,
            "context": form.context,
            "prediction": gender
        }
        return templates.TemplateResponse(
            "usvisa.html", 
            {"request": request, "result_gender": response_payload, "active_tab": "gender"}
        )
    except Exception as e:
        return templates.TemplateResponse("usvisa.html", {"request": request, "error": str(e)})

@app.post("/recommend_hotel")
async def recommend_hotel(request: Request):
    try:
        form = DataForm(request)
        await form.get_hotel_data()
        
        # Simulated Recommendations
        recommendations = [
            {"name": "Simulated Hotel 1", "place": "Rio", "price": 200, "days": form.days, "pop": "High"},
            {"name": "Simulated Hotel 2", "place": "Mumbai", "price": 150, "days": form.days, "pop": "Medium"},
            {"name": "Simulated Hotel 3", "place": "London", "price": 300, "days": form.days, "pop": "Low"},
        ]
        
        response_payload = {
            "task": "hotel_recommendation",
            "selected_hotel": form.hotel_name,
            "preferences": {"days": form.days, "price": form.price_pref, "pop": form.pop_pref},
            "recommendations": recommendations
        }
        
        return templates.TemplateResponse(
            "usvisa.html", 
            {"request": request, "result_hotel": response_payload, "active_tab": "hotel"}
        )
    except Exception as e:
        return templates.TemplateResponse("usvisa.html", {"request": request, "error": str(e)})

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)