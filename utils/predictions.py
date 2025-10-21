import joblib
import numpy as np

def calculate_radar_area(data):
    # Normaliza o RPM
    rpm = data['rpm'] / 100
    speed = data['speed']
    throttle = data['throttle']
    engine = data['engine_load']

    values = [rpm, speed, throttle, engine]

    # Fórmula da área do polígono
    angle = 2 * np.pi / len(values)
    area = 0.5 * np.abs(np.dot(values, np.roll(values, 1)) * np.sin(angle))
    
    return area

def predict_fuel_type(dados):
    if "fuel_type" in dados:
        prob = 1.0
        return dados["fuel_type"], prob
    elif "ethanol_percentage" in dados:
        model = joblib.load("./models/ethanol_model_rf.pkl")
        X = [dados["ethanol_percentage"],
             dados["speed"],
             dados["rpm"],
             dados["engine_load"],
             dados["throttle"],
             dados["timing_advance"]]
        X = np.array(X).reshape(1, -1)
        fuel_type = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        if fuel_type == 1:
            fuel_type_str = "Gasoline"
        elif fuel_type == 0:
            fuel_type_str = "Ethanol"

        return fuel_type_str, prob
    # else:
    #     pass
    return "Gasoline", 1.0
        
def predict_city_highway(dados):
    model = joblib.load("./models/city_highway_rf.pkl")
    X = [dados["speed"],
         dados["rpm"],
         dados["engine_load"],
         dados["throttle"],
         dados["timing_advance"],
        #  dados["accel_magnitude"]
        1.0
         ]
    X = np.array(X).reshape(1, -1)
    city_highway = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return city_highway, prob