# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Chatbot API with Hugging Face Kimi-K2-Thinking",
    description="A FastAPI application to handle chatbot logic using Hugging Face Inference API with Zephyr-7b.",
    version="1.0.0",
)

# Configuration for Hugging Face Inference API
# Try both HF_TOKEN and HUGGING_FACE_API_TOKEN for compatibility
HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
HUGGING_FACE_MODEL = os.getenv("HUGGING_FACE_MODEL")

if not HUGGING_FACE_API_TOKEN:
    raise ValueError("HUGGING_FACE_API_TOKEN environment variable not set. Please create a .env file.")

# Initialize OpenAI-compatible client for Hugging Face
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HUGGING_FACE_API_TOKEN,
)

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

async def query_huggingface_model(messages: list[dict], max_tokens: int = 250, temperature: float = 0.7):
    """
    Sends a query to the Hugging Face Inference API using OpenAI-compatible client.
    Includes robust error handling for API calls.
    
    Args:
        messages: List of message dictionaries with "role" and "content" keys
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 to 2.0)
    
    Returns:
        The generated text response
    """
    try:
        completion = client.chat.completions.create(
            model=HUGGING_FACE_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        if not completion.choices or not completion.choices[0].message:
            raise HTTPException(
                status_code=500,
                detail="No response generated from the AI model."
            )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        error_message = str(e)
        print(f"Error querying Hugging Face API: {error_message}")
        
        # Handle specific error cases
        if "503" in error_message or "unavailable" in error_message.lower():
            raise HTTPException(
                status_code=503,
                detail="AI model is currently loading on Hugging Face or temporarily unavailable. Please try again in a moment."
            )
        elif "401" in error_message or "unauthorized" in error_message.lower():
            raise HTTPException(
                status_code=401,
                detail="Invalid API token. Please check your HF_TOKEN or HUGGING_FACE_API_TOKEN."
            )
        elif "timeout" in error_message.lower():
            raise HTTPException(
                status_code=504,
                detail="Hugging Face API request timed out."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get response from AI model: {error_message}"
            )


@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    Endpoint to process user messages using the Hugging Face AI model
    and return chatbot replies, maintaining conversation context.
    """
    # Generate a new conversation ID if not provided by the client
    # The validator already converts integers to strings, so we can use it directly
    conversation_id = request.conversation_id if request.conversation_id else str(uuid.uuid4())

    # Retrieve current conversation history for this ID, or start a new one
    current_conversation = conversation_store.get(conversation_id, [])

    # Add the current user message to the history
    current_conversation.append({"role": "user", "content": request.message})

    try:
        # Query the Hugging Face model using OpenAI-compatible client
        # The client handles message formatting automatically
        bot_reply = await query_huggingface_model(
            messages=current_conversation,
            max_tokens=250,
            temperature=0.7
        )

        # Clean up the response (remove any extra whitespace)
        bot_reply = bot_reply.strip() if bot_reply else "I'm having trouble generating a response right now. Please try again."

        # Add the bot's reply to the conversation history
        current_conversation.append({"role": "assistant", "content": bot_reply})

        # Store the updated conversation history back into our in-memory store
        conversation_store[conversation_id] = current_conversation

    except HTTPException as e:
        raise e  # Re-raise the HTTPException that originated from query_huggingface_model
    except Exception as e:
        print(f"An unexpected error occurred during chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    return ChatResponse(reply=bot_reply, conversation_id=conversation_id)

