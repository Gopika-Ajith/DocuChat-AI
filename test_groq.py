from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

print("KEY FOUND:", os.getenv("GROQ_API_KEY"))

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

print(response)
print(response.choices[0].message.content)