# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import requests
from difflib import get_close_matches
from data import training_data

app = FastAPI()

class RecommendationRequest(BaseModel):
    user_input: str

class RecommendationResponse(BaseModel):
    response: str

def find_closest_match(user_input: str) -> Optional[str]:
    """Encuentra la entrada de training_data más parecida al user_input."""
    inputs = [item["user_input"] for item in training_data]
    closest_matches = get_close_matches(user_input, inputs, n=1, cutoff=0.6)
    if closest_matches:
        matched_input = closest_matches[0]
        for item in training_data:
            if item["user_input"] == matched_input:
                return item["response"]
    return None

def get_local_model_response(user_input: str) -> str:
    """Función para obtener respuesta del modelo de lenguaje local de LMStudio."""
    try:
        response = requests.post(
            "http://127.0.0.1:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "llama-3.2-3b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.7,
                "max_tokens": 50,
                "stream": False
            }
        )
        return response.json().get("joke_response", {}).get("joke", "No tengo una respuesta en este momento.")
    except Exception as e:
        return f"Error al conectar con el modelo local: {str(e)}"

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    # Intenta encontrar una coincidencia en los datos de entrenamiento
    closest_response = find_closest_match(request.user_input)
    if closest_response:
        return RecommendationResponse(response=closest_response)
    
    # Si no se encuentra una coincidencia cercana, llama al modelo local
    fallback_response = get_local_model_response(request.user_input)
    return RecommendationResponse(response=fallback_response)

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de recomendaciones de productos"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
