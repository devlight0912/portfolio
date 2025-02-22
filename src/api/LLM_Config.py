import os
import logging
import re
from dotenv import load_dotenv

# Import both providers
from openai import AzureOpenAI
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Determine which provider to use
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "AZURE").upper()
logger.info(f"LLM Provider: {LLM_PROVIDER}")

# Validate required environment variables based on provider
if LLM_PROVIDER == "AZURE":
    REQUIRED_ENV_VARS = [
        "OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
        "OPENAI_API_KEY",
        "MODEL_NAME"
    ]
elif LLM_PROVIDER == "GEMINI":
    REQUIRED_ENV_VARS = [
        "GEMINI_API_KEY"
    ]
else:
    raise ValueError("Unsupported LLM_PROVIDER. Please set LLM_PROVIDER to 'AZURE' or 'GEMINI'.")

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

# Configure Gemini API if using Gemini
if LLM_PROVIDER == "GEMINI":
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Azure client when needed
_azure_client = None
def get_azure_client():
    global _azure_client
    if _azure_client is None:
        try:
            _azure_client = AzureOpenAI(
                azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_version=os.getenv("OPENAI_API_VERSION"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("AzureOpenAI client initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize AzureOpenAI client.")
            raise e
    return _azure_client

def get_completion_from_messages(
    system_message: str,
    user_message: str,
    model: str = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    n: int = 1,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0
) -> str:
    """
    Generate a completion response from the selected LLM provider based on the provided system and user messages.

    For AzureOpenAI, additional parameters are supported.
    For Gemini, only the temperature parameter is utilized.

    Returns:
        str: The generated response content.
    """
    logger.info(f"System message: {system_message}")
    logger.info(f"User message: {user_message}")
    logger.info(f"Selected LLM Provider: {LLM_PROVIDER}")

    if LLM_PROVIDER == "AZURE":
        if model is None:
            model = os.getenv("MODEL_NAME")
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_message}
        ]
        client = get_azure_client()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=n,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
            completion = response.choices[0].message.content
            logger.info("Completion fetched successfully from Azure OpenAI API.")
            return completion
        except Exception as e:
            logger.exception("Error while fetching completion from Azure OpenAI API.")
            raise e

    elif LLM_PROVIDER == "GEMINI":
        try:
            # Combine system and user messages
            combined_message = f"{system_message}\n\nUser Query: {user_message}"
            logger.info("=== INPUT ===")
            logger.info(f"Combined Message:\n{combined_message}")
            logger.info(f"Temperature: {temperature}")

            # Generate response using Gemini's model (defaulting to 'gemini-1.5-pro')
            model_instance = genai.GenerativeModel('gemini-1.5-pro')
            response = model_instance.generate_content(
                contents=combined_message,
                generation_config={"temperature": temperature}
            )

            logger.info("=== RAW OUTPUT ===")
            logger.info(f"Response Object: {response}")

            # Ensure text is a string and perform basic cleaning
            text = response.text if isinstance(response.text, str) else str(response.text)
            clean_text = re.sub(r'```json\n|\n```', '', text)

            logger.info("=== CLEANED OUTPUT ===")
            logger.info(f"Cleaned Text:\n{clean_text}")

            return clean_text
        except Exception as e:
            logger.exception(f"Error generating response from Gemini: {str(e)}")
            raise e
    else:
        raise ValueError("Unsupported LLM_PROVIDER specified.")
