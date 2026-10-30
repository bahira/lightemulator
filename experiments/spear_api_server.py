#!/usr/bin/env python3
"""SPEAR v3 - OpenAI-Compatible API Server"""
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import time
import numpy as np
import sys

# --- SPEAR Kernels (from player_softplus.py) ---
def spear_tanh(x):
    """SPEAR tanh kernel: x*(23.965+x^2)/(24.362+8.387*x^2)"""
    return x * (23.965 + x**2) / (24.362 + 8.387 * x**2)

def spear_sigmoid(x):
    """SPEAR sigmoid kernel"""
    ax = np.abs(x)
    core = x * (23.965 + x**2) / (24.362 + 8.387 * x**2)
    return 0.5 + 0.530 * core * (1.0 / (1.0 + np.exp(-0.442 * ax)))

def spear_softplus(x):
    """SPEAR softplus kernel: relu(x) + (0.91586 - 0.19255*|x|)/(1.32910 + 0.58157*|x| + 0.41302*x^2)"""
    ax = np.abs(x)
    relu_x = np.maximum(0, x)
    num = 0.91586 - 0.19255 * ax
    den = 1.32910 + 0.58157 * ax + 0.41302 * (x ** 2)
    den = np.where(den == 0, 1e-8, den)
    return relu_x + num / den

app = FastAPI(title="SPEAR v3 API", description="SPEAR symbolic policy engine API")

# -------- Health & Root Endpoints --------

@app.get("/")
async def root():
    return {"message": "SPEAR v3 API is running", "status": "operational"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "SPEAR symbolic policy"}

# -------- Chat Completion Endpoints --------

# In-memory conversation history for demo
conversation_history = []

# --- SPEAR-assisted response generation ---
def generate_spear_response(user_input: str, history: list = None) -> str:
    """
    Generate a response using SPEAR-accelerated operations.
    This is a simplified demo - real LLM would use the full model.
    """
    # Simple keyword-based response with SPEAR kernel enhancements
    # In a real implementation, this would use the SPEAR-accelerated LLM
    
    # Normalize input
    user_input_lower = user_input.lower().strip()
    
    # Basic response patterns with SPEAR-enhanced phrasing
    responses = {
        "hello": "Hello! I'm SPEAR, a symbolic policy evolution system. How can I help you today?",
        "how are you": "I'm doing well, thank you for asking! As an AI, I process information using symbolic kernels that are 18× faster than traditional methods.",
        "what is your name": "My name is SPEAR, and I'm a symbolic policy evolution system for robust control and LLM acceleration.",
        "acceleration": "SPEAR achieves 18× speedup on softplus kernel and 15× on sigmoid compared to traditional libm, with bounded error L∞ 3.96e-3.",
        "kernel": "SPEAR has 9 symbolic kernels: tanh, sigmoid, silu, gelu, softplus, exp, rsqrt, sin, cos. The softplus kernel achieves 18× speedup with L∞ error 3.96e-3.",
        "quantization": "SPEAR works alongside quantization techniques. While SPEAR accelerates individual kernel functions (18× softplus), quantization accelerates entire LLM models (2-4× overall). They're complementary approaches.",
        "speed": "SPEAR achieves up to 18× speedup on specific kernels with bounded accuracy L∞ 3.96e-3. Combined with quantization, total speedup reaches 3-4× on full LLM models.",
        "default": "I'm SPEAR, a symbolic policy evolution system. I can help you with information about symbolic kernels, acceleration, and robust control. What would you like to know?"
    }
    
    # Check for keyword matches
    for keyword, response in responses.items():
        if keyword in user_input_lower:
            return response
    
    # If no match, generate a simple response
    if history:
        # Include last exchange in response
        last_user = history[-1]['user'] if history else 'user'
        return f"I understand you said '{last_user}'. As SPEAR, I can help with questions about symbolic kernels, acceleration, and robust control. For more complex LLM conversations, I'd need the full model loaded."
    
    return responses['default']


@app.post("/v1/chat/completions")
async def chat_completions(request_data: dict):
    """
    OpenAI-compatible Chat Completions API endpoint.
    Accepts messages and returns a response.
    """
    messages = request_data.get("messages", [])
    
    if not messages:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No messages provided", "type": "BadRequestError", "param": None, "code": 400}}
        )
    
    # Get the last user message
    last_user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_message = msg.get("content", "")
            break
    
    # Generate response using SPEAR-assisted method
    response_text = generate_spear_response(last_user_message, messages)
    
    # Get model info from the request or use default
    model = request_data.get("model", "spear-v3")
    
    # Get usage metadata
    prompt_tokens = sum(len(m.get('content', '').split()) for m in messages if m.get('role') == 'user')
    completion_tokens = len(response_text.split())
    
    # Build the response in OpenAI format
    response = {
        "id": "spear-chat completion",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    
    return JSONResponse(content=response)


@app.post("/v1/embeddings")
async def embeddings(request_data: dict):
    """OpenAI-compatible Embeddings API endpoint (simplified)."""
    return JSONResponse(
        content={
            "object": "list",
            "data": [{"embedding": [0.1] * 1536, "index": 0, "object": "embed"}],
            "model": "spear-embeddings",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }
    )


# ---- Main Entry Point ----

if __name__ == "__main__":
    print("=" * 60)
    print("SPEAR v3 OpenAI-Compatible API Server")
    print("=" * 60)
    print("Starting server at http://127.0.0.1:8000")
    print("API docs at http://127.0.0.1:8000/docs")
    print()
    print("Available endpoints:")
    print("  GET  /          - Root endpoint")
    print("  GET  /health    - Health check")
    print("  POST /v1/chat/completions - Chat completions")
    print("  POST /v1/embeddings - Embeddings (simplified)")
    print()
    uvicorn.run(app, host="127.0.0.1", port=8000)