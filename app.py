from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response, HTMLResponse
# Import run_in_threadpool to fix the timeout/blocking issue
from fastapi.concurrency import run_in_threadpool 
from uvicorn import run as app_run
from typing import Optional
import pandas as pd
from datetime import datetime
import json

# Assuming these exist in your project structure
from voyage_analytics.constants import APP_HOST, APP_PORT
from voyage_analytics.entity.local_estimator import Local_Estimator_Class

# --- SETUP ---
# Initialize the model once. If this takes a long time, the server won't start immediately.
print("Loading Model...") 
local_estimator = Local_Estimator_Class()  
print("Model Loaded.")

app = FastAPI()

templates = Jinja2Templates(directory='templates')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPER CLASSES (Kept same as your code) ---
class FlightData:
    def __init__(self, from_loc, to_loc, flight_type, time, distance, agency, date):
        self.from_loc = from_loc
        self.to_loc = to_loc
        self.flight_type = flight_type
        self.time = time
        self.distance = distance
        self.agency = agency
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            self.day = dt.strftime("%A")
        except:
            self.day = "Monday"

    def get_data_as_dataframe(self):
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

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        # ... (Your existing attributes) ...
        self.travel_code: Optional[int] = None
        self.from_loc: Optional[str] = None
        self.to_loc: Optional[str] = None
        self.flight_type: Optional[str] = None
        self.price: Optional[float] = None
        self.time: Optional[float] = None
        self.distance: Optional[float] = None
        self.agency: Optional[str] = None
        self.date: Optional[str] = None
        self.full_name: Optional[str] = None
        self.context: Optional[str] = None
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
        self.hotel_name = form.get("hotelSelect", "").split(',')[0]
        self.days = int(form.get("stayDays", 1))
        self.price_pref = form.get("pricePref", "Any")
        self.pop_pref = form.get("popPref", "Any")


# --- ROUTES ---

@app.get("/", tags=["authentication"], response_class=HTMLResponse)
async def index(request: Request):
    # Ensure your HTML file is named 'usvisa.html' and is inside a 'templates' folder
    return templates.TemplateResponse("usvisa.html", {"request": request, "context": "Rendering"})

@app.post("/predict_flight")
async def predict_flight(request: Request):
    try:
        form = DataForm(request)
        await form.get_flight_data()
        
        flight_data = FlightData(
            from_loc=form.from_loc,
            to_loc=form.to_loc,
            flight_type=form.flight_type,
            time=form.time,
            distance=form.distance,
            agency=form.agency,
            date=form.date
        )
        
        df = flight_data.get_data_as_dataframe()
        
        # --- FIX: Use run_in_threadpool for blocking ML inference ---
                
        
        prediction_result = local_estimator.regression_predict_func(df)[0]
        prediction_result = float(prediction_result)

        response_payload = {
            "task": "flight_price_prediction",
            "input_received": df.to_dict(),
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
        
        df = pd.DataFrame({'name': [form.full_name]})
        
        # --- FIX: Use run_in_threadpool ---
        gender = str(local_estimator.classification_predict_func(df)[0])
        
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
        
        # --- FIX: Use run_in_threadpool ---
        # Note: We must pass the function and arguments separately
        recommendations_df = local_estimator.recommendation_predict_func(form.hotel_name)
        
        # orient="records" makes it easy to loop in HTML
        recommendations = recommendations_df.reset_index(drop=True).to_dict(orient="records")
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
    # Ensure Host is 0.0.0.0 to listen on all interfaces (fixes connection timeouts)
    # If APP_HOST is imported, make sure it is valid, otherwise default to "0.0.0.0"
    
    port = int(APP_PORT) if APP_PORT else 8000
    app_run(app, host='127.0.0.1', port=port)