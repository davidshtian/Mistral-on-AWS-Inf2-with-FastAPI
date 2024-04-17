from typing import Dict
from fastapi import FastAPI, Request, HTTPException
from backend_model import MistralModel

app = FastAPI()
mistral_model = MistralModel()

@app.post("/generate")
async def generate(request: Request) -> Dict[str, str]:
    """
    Generate text using the Mistral model.

    Args:
        request (Request): The incoming request object.

    Returns:
        Dict[str, str]: A dictionary containing the generated text or an error message.
    """
    try:
        request_payload = await request.json()
        inputs = request_payload.get("inputs")
        parameters = request_payload.get("parameters", {})

        if not inputs:
            raise HTTPException(status_code=400, detail="No input provided")

        generated_text = mistral_model.generate(inputs, parameters)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
