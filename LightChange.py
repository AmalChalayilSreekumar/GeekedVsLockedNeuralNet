import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOVEE_API_KEY")
DEVICE1_ID = os.getenv("DEVICE1_ID")
DEVICE2_ID = os.getenv("DEVICE2_ID")
MODEL = os.getenv("MODEL")

HEADERS = {
    "Govee-API-Key": API_KEY,
    "Content-Type": "application/json"
}

url = "https://developer-api.govee.com/v1/devices/control"

def changeRed(DEVICE_ID):
    '''
    Changes GOVEE lightbulb to Red (150)
    
    Args:
        DEVICE_ID: Takes in the device id/mac address of the light desired to be changed
    
    Return:
        JSON of return what is return from API call
    '''
    payload = {
        "device": DEVICE_ID,
        "model": MODEL,
        "cmd": {
            "name": "color",
            "value": {"r": 150,"g": 0,"b": 0}
        }
    }
    response = requests.put(url, headers=HEADERS, json=payload)
    return response.json()

def changeWhite(DEVICE_ID):
    '''
    Changes GOVEE lightbulb to white (max)
    
    Args:
        DEVICE_ID: Takes in the device id/mac address of the light desired to be changed
    
    Return:
        JSON of return what is return from API call
    '''
    payload = {
        "device": DEVICE_ID,
        "model": MODEL,
        "cmd": {
            "name": "color",
            "value": {"r": 244,"g": 244,"b": 244}
        }
    }
    response = requests.put(url, headers=HEADERS, json=payload)
    print(response.json())


# changeRed(DEVICE1_ID)
# changeRed(DEVICE2_ID)

# changeWhite(DEVICE1_ID)
# changeWhite(DEVICE2_ID)
