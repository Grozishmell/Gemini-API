from google import genai
from google.genai import types


client = genai.Client(api_key="YOUR_API_KEY")

chat = client.chats.create(
    model = "gemini-3.5-flash",
    config = types.GenerateContentConfig(
        system_instructions = "Ты - строгий, но добрый учитель. Отвечай кратко."
    )
)

print("Чат запущен. Напиши 'выход', чтобы закончить.\n")

while True:
    user_input = input("Ты: ")
    if user_input.lower() == "выход":
        break
    user_input = chat.send_message(user_input)
    print("Gemini:", user_input.text, "\n")