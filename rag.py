import numpy as np
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types



# Читаем PDF и достаем весь текст
reader = PdfReader("Резюме Баев Н.В. Python разработчик.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

print(f"Всего символов в документе: {len(full_text)}\n")


# Режем текст на кусочки
def split_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap # сдвигаемся назад на overlap, чтобы не разрывать смысл
    return chunks

chunks = split_into_chunks(full_text)


client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Превращаем все кусочки в векторы
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=chunks,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
)
chunk_vectors = [e.values for e in result.embeddings]
print(f"Готово: каждый кусочек - это вектор из {len(chunk_vectors[0])} чисел\n")


# Находим кусочки, подходящие к вопросу
def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_chunks(question, top_k=3):
    # Превращаем вопрос в вектор
    q = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    q_vector = q.embeddings[0].values

    # считаем близость вопроса к каждому кусочку
    scores = [cosine(q_vector, v) for v in chunk_vectors]
    # берем индексы top_k самых близких
    best = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in best]


# Получаем готовый ответ на основе найденных кусочков
def answer_question(question):
    # Находим  подходящие кусочки
    relevant = find_relevant_chunks(question)
    context = "\n\n".join(relevant)

    # Строим промпт: даем контекст и строгую инструкцию
    prompt = f"""Ты - помощник, отвечающий на вопросы по документу.
Используй только инфомацию из контекста ниже.
Если ответа в контексте нет - честно скажи: "В документе этого нет."

Контекст:
{context}

Вопрос: {question}
"""
    
    # Отдаем модели
    response = client.models.generate_content(
        model = "gemini-3.5-flash",
        contents = prompt,
    )
    return response.text

# Проверка
question = "Какой рост у Баева Н.В.?"
print("Вопрос:", question)
print("\nОтвет:", answer_question(question))