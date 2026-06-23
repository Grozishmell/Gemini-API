import streamlit as st
import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import types

st.title("База знаний")


# Создание ресурсов
@st.cache_resource
def get_clients():
    gemini = genai.Client(api_key = st.secrets["GEMINI_API_KEY"])
    db = chromadb.PersistentClient(path = "./knowledge_db")
    collection = db.get_or_create_collection(name = "my_knowledge")
    return gemini, collection

gemini, collection = get_clients()


def split_into_chunks(text, chunk_size = 1000, overlap = 200):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def embed(texts, task):
    result = gemini.models.embed_content(
        model = "gemini-embedding-001",
        contents = texts,
        config = types.EmbedContentConfig(task_type = task),
    )
    return [e.values for e in result.embeddings]


def add_pdf(file):
    name = file.name
    if collection.get(where = {"source": name})["ids"]:
        return False # уже в базе
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    chunks = split_into_chunks(text)
    vectors = embed(chunks, "RETRIEVAL_DOCUMENT")
    collection.add(
        documents = chunks,
        embeddings = vectors,
        ids = [f"{name}_{i}" for i in range(len(chunks))],
        metadatas = [{"source": name} for _ in chunks],
    )
    return True


def ask(question):
    q_vector = embed(question, "RETRIEVAL_QUERY")[0]
    results = collection.query(query_embeddings = [q_vector], n_results = 3)
    context = "\n\n".join(results["documents"][0])
    sources = {m["source"] for m in results["metadatas"][0]}
    prompt = f""" Ты - помощник по базе знаний.
Используй ТОЛЬКО контекст ниже. Если ответа нет - скажи: "В базе этого нет."PermissionError

Контекст:
{context}

Вопрос: {question}
"""
    response = gemini.models.generate_content(
        model = "gemini-3.5-flash", contents = prompt
    )
    return response.text, sources


# Загрузка: можно сразу несколько файлов
if "processed" not in st.session_state:
    st.session_state.processed = set()

uploaded = st.file_uploader("Добавь PDF в базу", type = "pdf", accept_multiple_files = True)
if uploaded:
    for file in uploaded:
        if file.name in st.session_state.processed:
            continue # этот файл уже обрабатывали в этой сессии
        with st.spinner(f"Обрабатываю {file.name}..."):
            added = add_pdf(file)
        st.success(f"Добавлен: {file.name}" if added else st.info(f"{file.name} уже в базе"))
        st.session_state.processed.add(file.name)


# Список документов в базе(боковая панель)
in_db = sorted({m["source"] for m in collection.get()["metadatas"]})
st.sidebar.header("В базе сейчас")
if in_db:
    for s in in_db:
        st.sidebar.write("•", s)
else:
    st.sidebar.info("База пуста")


# Вопрос
if in_db:
    question = st.text_input("Задай вопрос по базе знаний:")
    if question:
        with st.spinner("Ищу ответ..."):
            answer, sources = ask(question)
        st.markdown("### Ответ")
        st.write(answer)
        st.caption("Источники: " + ", ".join(sources))
else:
    st.info("Загрузи хотя бы один PDF, чтобы задавать вопросы")