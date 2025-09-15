from typing import Optional

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = os.environ.get("MODEL_ID", "Shresthadev403/food-recipe-generation")


class GenerateRequest(BaseModel):
	prompt: Optional[str] = Field(
		default=None,
		description="Free-form prompt. If omitted, a prompt will be constructed from the provided fields.",
	)
	ingredients: Optional[str] = Field(
		default=None, description="Comma-separated list of ingredients to include"
	)
	cuisine: Optional[str] = Field(default=None, description="Cuisine preference, e.g., Italian")
	dietary: Optional[str] = Field(
		default=None, description="Dietary preference, e.g., vegetarian, vegan, gluten-free"
	)
	max_new_tokens: int = Field(default=256, ge=16, le=1024)
	temperature: float = Field(default=0.8, ge=0.0, le=2.0)
	top_p: float = Field(default=0.95, ge=0.1, le=1.0)
	do_sample: bool = Field(default=True)


class GenerateResponse(BaseModel):
	recipe: str
	model_id: str
	used_prompt: str


app = FastAPI(title="Recipe Generation API", version="1.0.0")


tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForCausalLM] = None
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_prompt(request: GenerateRequest) -> str:
	if request.prompt:
		return request.prompt.strip()

	parts = [
		"You are a helpful chef. Generate a clear, well-structured recipe.",
	]
	if request.ingredients:
		parts.append(f"Ingredients to use: {request.ingredients}.")
	if request.cuisine:
		parts.append(f"Cuisine: {request.cuisine}.")
	if request.dietary:
		parts.append(f"Dietary preferences: {request.dietary}.")

	parts.append(
		"Return sections: Title, Ingredients (bulleted with quantities), Instructions (numbered), and Tips."
	)
	return " \n".join(parts).strip()


@app.on_event("startup")
def load_model() -> None:
	global tokenizer, model
	try:
		tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
		model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
		model.to(device)
		model.eval()
	except Exception as exc:
		raise RuntimeError(f"Failed to load model {MODEL_ID}: {exc}") from exc


@app.get("/health")
def health() -> dict:
	loaded = tokenizer is not None and model is not None
	return {"status": "ok" if loaded else "loading", "model_id": MODEL_ID, "device": str(device)}


@app.post("/generate", response_model=GenerateResponse)
def generate_recipe(request: GenerateRequest) -> GenerateResponse:
	if tokenizer is None or model is None:
		raise HTTPException(status_code=503, detail="Model is not loaded yet")

	user_prompt = build_prompt(request)
	inputs = tokenizer(user_prompt, return_tensors="pt").to(device)

	with torch.no_grad():
		output_ids = model.generate(
			**inputs,
			max_new_tokens=request.max_new_tokens,
			temperature=request.temperature,
			top_p=request.top_p,
			do_sample=request.do_sample,
			pad_token_id=tokenizer.eos_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)

	generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

	return GenerateResponse(
		recipe=generated_text,
		model_id=MODEL_ID,
		used_prompt=user_prompt,
	)


@app.get("/")
def serve_index() -> FileResponse:
	index_path = os.path.join(os.path.dirname(__file__), "index.html")
	if not os.path.exists(index_path):
		raise HTTPException(status_code=404, detail="index.html not found")
	return FileResponse(index_path, media_type="text/html")


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)


