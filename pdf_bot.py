import streamlit as st
import numpy as np
from pypdf import PdfReader
from google import genai
from google.genai import types


st.title("Бот по PDF документу")

if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
client = st.session_state.client


def split_into_chunks(text, chunk_size=1000, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Загрузка PDF
uploaded_file = st.file_uploader("Загрузи PDF", type="pdf")

# Обработка документа только когда загружен новый файл
if uploaded_file and uploaded_file.name != st.session_state.get("current_file"):
    with st.spinner("Читаю и обрабатываю документ..."):
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            full_text += (page.extract_text() or "") + "\n"

        chunks = split_into_chunks(full_text)

        # превращаем в векторы один раз и кладем в память
        result = client.models.embed_content(
            model = "gemini-embedding-001",
            contents = chunks,
            config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        st.session_state.chunks = chunks
        st.session_state.chunk_vectors = [e.values for e in result.embeddings]
        st.session_state.current_file = uploaded_file.name
    st.success(f"Документ готов! Кусочков: {len(chunks)}")

# Вопрос и ответ
if "chunks" in st.session_state:
    question = st.text_input("Задай вопрос по документу:")
    if question:
        with st.spinner("Думаю..."):
            q = client.models.embed_content(
                model = "gemini-embedding-001",
                contents = question,
                config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            q_vector = q.embeddings[0].values

            scores = [cosine(q_vector, v) for v in st.session_state.chunk_vectors]
            best = np.argsort(scores)[-3:][::-1]
            context = "\n\n".join(st.session_state.chunks[i] for i in best)

            prompt = f"""Ты - помощник, отвечающий на вопросы по документу.
Используй ТОЛЬКО информацию из контекста ниже.
Если ответа в контексте нет - честно скажи: "В документе этого нет."

Контекст:
{context}

Вопрос: {question}
"""
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
            )
        st.markdown("### Ответ:")
        st.write(response.text)
else:
    st.info("Сначала загрузи PDF выше")