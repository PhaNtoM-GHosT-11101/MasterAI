import os
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI, AsyncOpenAI
import requests
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not GEMINI_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in .env")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in .env")
if not PERPLEXITY_API_KEY:
    logger.warning("PERPLEXITY_API_KEY not found in .env")


# --- Base Client ---
class AIClient(ABC):
    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# --- Implementations ---

class GeminiClient(AIClient):
    def __init__(self):
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    @property
    def name(self) -> str:
        return "Gemini"

    async def generate_response(self, prompt: str) -> str:
        if not self.model:
            return "Error: Gemini API Key missing."
        try:
            # Gemini async generation is a bit different, often standard call is synchronous enough or use run_in_executor if needed.
            # Newer SDKs have generate_content_async. Let's try to use it if available, else synchronous in a thread.
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return f"Error generating response from Gemini: {e}"


class OpenAIClient(AIClient):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    @property
    def name(self) -> str:
        return "ChatGPT"

    async def generate_response(self, prompt: str) -> str:
        if not self.client:
            return "Error: OpenAI API Key missing."
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI Error: {e}")
            return f"Error generating response from ChatGPT: {e}"


class PerplexityClient(AIClient):
    def __init__(self):
        self.api_key = PERPLEXITY_API_KEY
        self.base_url = "https://api.perplexity.ai/chat/completions"

    @property
    def name(self) -> str:
        return "Perplexity"

    async def generate_response(self, prompt: str) -> str:
        if not self.api_key:
            return "Error: Perplexity API Key missing."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.1-sonar-large-128k-online", 
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            # Running requests in a thread since it's blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: requests.post(self.base_url, json=data, headers=headers))
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"Perplexity Error: {response.status_code} - {response.text}")
                return f"Error generating response from Perplexity: Status {response.status_code}"
        except Exception as e:
            logger.error(f"Perplexity Error: {e}")
            return f"Error generating response from Perplexity: {e}"


# --- Judge & Curator ---

class MasterJudge:
    def __init__(self):
        # We'll use Gemini 1.5 Pro (or the strongest available) as the judge due to large context
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.judge_model = genai.GenerativeModel('gemini-1.5-pro')
        else:
            self.judge_model = None

    async def judge_and_curate(self, original_prompt: str, responses: Dict[str, str]) -> str:
        if not self.judge_model:
            return "Error: Judge model (Gemini) not available."

        # Construct the context for the judge
        context = f"""
You are the MasterAI Judge. Your task is to evaluate responses from multiple AI models to a user's prompt and synthesize the absolute best, most accurate, and comprehensive answer.

User Prompt: "{original_prompt}"

---
Response from Gemini:
{responses.get('Gemini', 'N/A')}

---
Response from ChatGPT:
{responses.get('ChatGPT', 'N/A')}

---
Response from Perplexity:
{responses.get('Perplexity', 'N/A')}

---

INSTRUCTIONS:
1. Analyze each response for accuracy, depth, clarity, and relevance.
2. Identify the strengths and weaknesses of each.
3. CONSTRUCT A FINAL, CURATED RESPONSE that combines the best parts of all three. Do not just list the responses. SYNTHESIZE them into one coherent, superior answer.
4. After the synthesized answer, provide a brief "Judge's Rationale" explaining why you chose certain elements and which model performed best.
"""
        try:
            result = await self.judge_model.generate_content_async(context)
            return result.text
        except Exception as e:
            logger.error(f"Judge Error: {e}")
            return f"Error during judging process: {e}"


# --- Main Application Logic ---

async def main():
    print(Fore.CYAN + Style.BRIGHT + "=========================================")
    print(Fore.CYAN + Style.BRIGHT + "      MasterAI - The Integrated Hive Mind")
    print(Fore.CYAN + Style.BRIGHT + "=========================================\n")

    clients = [GeminiClient(), OpenAIClient(), PerplexityClient()]
    judge = MasterJudge()

    while True:
        try:
            user_input = input(Fore.YELLOW + "Enter your prompt (or 'exit' to quit): " + Style.RESET_ALL)
            if user_input.lower() in ['exit', 'quit']:
                print(Fore.CYAN + "Goodbye!")
                break

            if not user_input.strip():
                continue

            print(Fore.BLUE + "\n[MasterAI] Querying the Council of AIs...")
            
            # Run all clients in parallel
            tasks = [client.generate_response(user_input) for client in clients]
            results = await asyncio.gather(*tasks)

            responses = {}
            for client, result in zip(clients, results):
                responses[client.name] = result
                print(Fore.GREEN + f" -> {client.name} has responded.")
                # Optional: Print individual responses for debugging? 
                # print(f"\n[{client.name}]: {result[:100]}...\n") 

            print(Fore.BLUE + "\n[MasterAI] The Judge is deliberating and curating the best answer...\n")
            
            final_verdict = await judge.judge_and_curate(user_input, responses)

            print(Fore.WHITE + Style.BRIGHT + "================ FINAL CURATED RESPONSE ================\n")
            print(final_verdict)
            print(Fore.WHITE + Style.BRIGHT + "\n======================================================\n")

        except KeyboardInterrupt:
            print(Fore.CYAN + "\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(Fore.RED + f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())