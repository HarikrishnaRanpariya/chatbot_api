# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Chatbot API with Hugging Face Zephyr-7b",
    description="A FastAPI application to handle chatbot logic using Hugging Face Inference API with Zephyr-7b.",
    version="1.0.0",
)

# Configuration for Hugging Face Inference API
HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
# Specify the Zephyr-7b model
HUGGING_FACE_MODEL = "HuggingFaceH4/zephyr-7b-beta" # Or "HuggingFaceH4/zephyr-7b-gemma-v0.1" for a different variant

if not HUGGING_FACE_API_TOKEN:
    raise ValueError("HUGGING_FACE_API_TOKEN environment variable not set. Please create a .env file.")

# In-memory store for conversation history (WARNING: NOT suitable for production)
# This will lose data if the server restarts and does not scale.
# For production, consider a database (e.g., Redis, PostgreSQL, MongoDB)
# Key: conversation_id (str), Value: List of {"role": "user" | "assistant", "content": "message"}
conversation_store = {}

# Define the request body model
class ChatRequest(BaseModel):
    message: str
    user_id: str = None # Optional user ID
    conversation_id: str = None # Optional: client can send to continue a conversation

# Define the response body model
class ChatResponse(BaseModel):
    reply: str
    conversation_id: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Chatbot API! Visit /docs for API documentation."}

def apply_zephyr_chat_template(messages: list[dict]) -> str:
    """
    Applies the Zephyr-7b chat template (Mistral-like) to a list of messages.
    The template format is:
    <s>[INST] User message 1 [/INST] Assistant response 1 </s>
    <s>[INST] User message 2 [/INST]
    """
    formatted_prompt = ""
    for message in messages:
        if message["role"] == "user":
            formatted_prompt += f"<s>[INST] {message['content']} [/INST]"
        elif message["role"] == "assistant":
            # Note the space before assistant's content and the </s> token
            formatted_prompt += f" {message['content']} </s>"
    return formatted_prompt

async def query_huggingface_model(payload: dict):
    """
    Sends a query to the Hugging Face Inference API and returns the response.
    Includes robust error handling for API calls.
    """
    API_URL = f"https://api-inference.huggingface.co/models/{HUGGING_FACE_MODEL}"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"}

    try:
        # requests.post is synchronous, but FastAPI's async functions can manage
        # I/O-bound tasks reasonably well for moderate loads. For very high concurrency,
        # consider an async HTTP client like 'httpx' and 'await client.post(...)'.
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        # Handle specific HTTP errors from the Hugging Face API
        if response.status_code == 503:
            raise HTTPException(
                status_code=503,
                detail="AI model is currently loading on Hugging Face or temporarily unavailable. Please try again in a moment."
            )
        print(f"HTTP error querying Hugging Face API: {http_err} - Response: {response.text}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Hugging Face API returned an error: {response.text}"
        )
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error querying Hugging Face API: {conn_err}")
        raise HTTPException(status_code=500, detail="Could not connect to Hugging Face API. Please check your network.")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error querying Hugging Face API: {timeout_err}")
        raise HTTPException(status_code=504, detail="Hugging Face API request timed out.")
    except requests.exceptions.RequestException as req_err:
        print(f"An unexpected request error occurred with Hugging Face API: {req_err}")
        raise HTTPException(status_code=500, detail=f"Failed to get response from AI model: {req_err}")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    Endpoint to process user messages using the Hugging Face Zephyr-7b AI model
    and return chatbot replies, maintaining conversation context.
    """
    # Generate a new conversation ID if not provided by the client
    conversation_id = request.conversation_id if request.conversation_id else str(uuid.uuid4())

    # Retrieve current conversation history for this ID, or start a new one
    current_conversation = conversation_store.get(conversation_id, [])

    # Add the current user message to the history
    current_conversation.append({"role": "user", "content": request.message})

    # Apply the Zephyr chat template to the entire conversation history.
    # This prepares the prompt string that the model will use to generate the next assistant response.
    full_prompt_for_generation = apply_zephyr_chat_template(current_conversation)

    # Prepare payload for Hugging Face Inference API
    payload = {
        "inputs": full_prompt_for_generation,
        "parameters": {
            "max_new_tokens": 250,      # Max length of the generated response
            "temperature": 0.7,         # Controls randomness: higher = more random, lower = more focused
            "do_sample": True,          # Use sampling for generation
            "top_p": 0.9,               # Nucleus sampling: consider tokens with cumulative probability up to top_p
            "repetition_penalty": 1.1,  # Penalize repeated tokens to avoid loops
            "return_full_text": False,  # Important: Only return the newly generated text, not the prompt
        },
        "options": {
            "wait_for_model": True      # Wait if the model is loading on Hugging Face's infrastructure
        }
    }

    try:
        hf_response = await query_huggingface_model(payload)

        if not hf_response or not isinstance(hf_response, list) or not hf_response[0].get("generated_text"):
            bot_reply = "I'm having trouble generating a response from Zephyr-7b right now. Please try again."
        else:
            raw_generated_text = hf_response[0]["generated_text"].strip()
            # Zephyr-7b might sometimes generate additional tokens like '</s>' or leading/trailing spaces.
            # We clean it up to get a clean response.
            bot_reply = raw_generated_text.replace("</s>", "").strip()

        # Add the bot's reply to the conversation history
        current_conversation.append({"role": "assistant", "content": bot_reply})

        # Store the updated conversation history back into our in-memory store
        conversation_store[conversation_id] = current_conversation

    except HTTPException as e:
        raise e # Re-raise the HTTPException that originated from query_huggingface_model
    except Exception as e:
        print(f"An unexpected error occurred during chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    return ChatResponse(reply=bot_reply, conversation_id=conversation_id)

