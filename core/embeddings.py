from langchain_google_genai import ChatGoogleGenerativeAI
import os
from core.config import settings

def get_embedding_model() -> ChatGoogleGenerativeAI:
    """Get an instance of a Google Generative AI embedding model"""

    return ChatGoogleGenerativeAI(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )
