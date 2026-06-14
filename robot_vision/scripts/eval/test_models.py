import os
from google import genai

api_key = "AIzaSyBYbAm1603zmMl6tu7WIJjORuFKrzAiLqo"
client = genai.Client(api_key=api_key)

for model in client.models.list():
    print(model.name)
