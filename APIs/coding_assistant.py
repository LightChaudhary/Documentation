from openai import OpenAI


user_input = input("How can I help you?")
client = OpenAI()

code_response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
            "role" : "system",
            "content": (
                "You are a Python coding assistant."
                "Only answer Python-related questions"
            ),
        },
        {
            "role": "user",
            "content": f"{user_input}",
        },
    ],
)

print(f"\n{code_response.output_text}")