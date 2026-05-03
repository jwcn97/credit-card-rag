import os
from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


def get_collection_name() -> str:
    return os.getenv("COLLECTION_NAME", "index").strip()


def get_persist_directory() -> str:
    return os.getenv("PERSIST_DIRECTORY", "./db/index").strip()


@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))


@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name=get_collection_name(),
        embedding_function=get_embeddings(),
        persist_directory=get_persist_directory(),
    )


def get_retriever():
    return get_vector_store().as_retriever(search_kwargs={"k": 4})


@st.cache_resource
def get_chat_model():
    return init_chat_model(os.getenv("CHAT_MODEL"))


def build_prompt(question: str, context: str) -> str:
    return f"""You are a helpful assistant answering questions based on the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:"""


def ingest_pdf_urls(urls: list[str]) -> tuple[int, list[str]]:
    all_chunks = []
    errors = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            loader = PyPDFLoader(url)
            data = loader.load()
            chunks = splitter.split_documents(data)
            all_chunks.extend(chunks)
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    if all_chunks:
        get_vector_store().add_documents(documents=all_chunks)

    return len(all_chunks), errors


def clear_current_collection() -> int:
    """
    Delete all documents in the active collection without removing the DB files.
    Returns the number of deleted chunks.
    """
    store = get_vector_store()
    # Read existing ids first so we can report how many chunks were removed.
    existing = store._collection.get(include=[])
    ids = existing.get("ids", [])
    if ids:
        store._collection.delete(ids=ids)
    return len(ids)


st.set_page_config(page_title="FAQ Chat", page_icon="💬")
st.title("💬 Ask Your Agent")
st.caption(
    f"Retrieval-Augmented Q&A over your data from `{get_collection_name()}`"
)

with st.sidebar:
    st.header("Data Management")

    with st.form("load_urls_form", clear_on_submit=True):
        urls_input = st.text_area(
            "PDF URLs (one per line)",
            placeholder="https://example.com/document.pdf",
        )
        load_urls_clicked = st.form_submit_button("Load URLs", use_container_width=True)

    if load_urls_clicked:
        urls = [u for u in urls_input.splitlines() if u.strip()]
        if not urls:
            st.warning("Please enter at least one PDF URL.")
        else:
            with st.spinner("Indexing documents into Chroma..."):
                chunk_count, errors = ingest_pdf_urls(urls)
            st.cache_resource.clear()
            st.toast(f"Loaded {len(urls)} URLs ({chunk_count} chunks).", icon="✅")
            if errors:
                st.error("Some URLs failed to load:")
                for err in errors:
                    st.write(f"- {err}")

    st.divider()
    st.subheader("Collection Management")

    if st.button("Delete all chunks (keep DB)", use_container_width=True):
        deleted_count = clear_current_collection()
        st.cache_resource.clear()
        st.toast(f"Deleted {deleted_count} chunks from `{get_collection_name()}`.", icon="✅")

query = st.text_input("Ask a question about your indexed documents:")

if query:
    retriever = get_retriever()
    model = get_chat_model()
    docs = retriever.invoke(query)

    if not docs:
        st.warning("No matching chunks found.")
    else:
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = build_prompt(query, context)
        response = model.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("Retrieved chunks"):
            for idx, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "unknown")
                st.markdown(f"**Chunk {idx}** — source: `{source}`, page: `{page}`")
                st.write(doc.page_content)
                st.divider()
