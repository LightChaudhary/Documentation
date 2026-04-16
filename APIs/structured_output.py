from openai import OpenAI
from pydantic import BaseModel
try:    
    client = OpenAI()

    class CodeOutput(BaseModel):    
        function_name : str
        code : str
        explanation : str
        example_usage : str

    response = client.responses.parse(
         model="gpt-4",
        input=[
            {
                "role" : "system",
                "content": "You are a coding assistant. Generate clean, well documented Python code"
            },
            {
                "role" : "user",
                "content": "Write a simple Python function to add two numbers."
            }
        ],
        text_format=CodeOutput,
    )

    result = response.output_parsed

    print(f"Function name: {result.function_name}")
    print("\nCode:")
    print(result.code)
    print(f"\nExplanation: {result.explanation}")
    print(f"\nExample Usage: {result.example_usage}")

except: 
    print("APIs limit reached.")