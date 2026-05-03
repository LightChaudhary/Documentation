import os
from dotenv import load_dotenv
from loguru import logger
import ollama

load_dotenv()

MODEL_NAME = os.getenv("LLM_MODEL", "qwen2.5-coder:latest")

def query_llm(prompt: str) -> str:
    "Call local LLM with error handling and structured logging."
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content":prompt}],
            options={"temperature":0.5, "num_predict":150}
        )
        return response["message"]["content"].strip()
    except ConnectionError:
        logger.error("Ollama serve not running. Start with: ollama serve")
        raise
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise 

def main():
    logger.info("Day 1: Local LLM integration initialized")

    test_prompt = "Explain RAG in one sentecnce for a non-technical founder."
    result = query_llm(test_prompt)

    logger.success(f"Response: {result}")
    logger.info("Next: wrap this in fastAPI + deploy")

if __name__ == "__main__":
    main()