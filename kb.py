import chromadb
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types


# Два клиента: Gemini для эмбеддингов и базы данных
gemini = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
db = chromadb.PersistentClient(path="./knowledge_db")
collection = db.get_or_create_collection(name="my_knowledge")


def split_into_chunks(text, chunk_size=1000, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def embed(texts, task):
    result = gemini.models.embed_content(
        model = "gemini-embedding-001",
        contents = texts,
        config = types.EmbedContentConfig(task_type=task),
    )
    return [e.values for e in result.embeddings]

# ИНДЕКСАЦИЯ: добавляем PDF в базу данных
def add_pdf(path):
    name = path.split("/")[-1]
    # если документ уже в базе - не индексируем заново
    if collection.get(where={"source": name})["ids"]:
        print(f"'{name}' уже в базе, пропускаю.")
        return
    
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    chunks = split_into_chunks(text)
    vectors = embed(chunks, "RETRIEVAL_DOCUMENT")

    # кладем в базу: тест, его вектор, уникальный id и метку "откуда"
    collection.add(
        documents=chunks,
        embeddings=vectors,
        ids=[f"{name}_{i}" for i in range(len(chunks))],
        metadatas=[{"source": name} for _ in chunks],
    )
    print(f"Добавлено '{name}': {len(chunks)} кусочков.")

# ВОПРОС: база сама ищет, модель отвечает
def ask(question):
    q_vector = embed(question, "RETRIEVAL_QUERY")[0]
    results = collection.query(query_embeddings=[q_vector], n_results=3)

    found_chunks = results["documents"][0]
    sources = results["metadatas"][0]
    context = "\n\n".join(found_chunks)

    prompt = f"""Ты - помощник по базе знаний.
Используй ТОЛЬКО контекст ниже. Если ответа нет - скажи: "В базе этого нет."

Контекст:
{context}

Вопрос: {question}
"""
    response = gemini.models.generate_content(
        model = "gemini-3.5-flash", contents = prompt
    )
    return response.text, sources


add_pdf("Вставь 1 pdf-файл")
add_pdf("Вставь 2 pdf-файл")
add_pdf("Вставь 3 pdf-файл")
print(f"\nВсего кусочков в базе: {collection.count()}\n")

answer, sources = ask("Какой опыт у Николая? Коля крутой? Коля умный?")
print("Ответ:", answer)
print("Источники:", {s["source"] for s in sources})