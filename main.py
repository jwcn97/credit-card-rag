import os
import requests
from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


CARD_URLS = {
    "CITI_REWARDS": "https://www.citibank.com.sg/credit-cards/rewards/citi-rewards-card/pdf/10x-rewards-promotion-terms-and-conditions-2020.pdf",
    "HSBC_REVOLUTION": "https://www.hsbc.com.sg/content/dam/hsbc/sg/documents/credit-cards/revolution/revo-up-promotion-terms-and-conditions.pdf",
    "DBS_WOMEN_WORLD": "https://www.dbs.com.sg/iwov-resources/media/pdf/cards/dbs-womans-card-tnc.pdf",
    "UOB_PPV": "https://www.uob.com.sg/web-resources/personal/pdf/personal/cards/credit-cards/rewards-cards/uob-preferred-platinum-visa-card/terms-and-conditions-for-preferred-plat-visa.pdf",
    "OCBC_REWARDS": "https://www.ocbc.com/iwov-resources/sg/ocbc/personal/pdf/cards/tnc-titaniumrewards-creditcard-programme-wef-1nov23.pdf",
}

CARD_ALIASES = {
    "CITI_REWARDS": "Citi Rewards Card",
    "HSBC_REVOLUTION": "HSBC Revolution Card",
    "DBS_WOMEN_WORLD": "DBS Women's World Card",
    "UOB_PPV": "UOB Preferred Platinum Visa",
    "OCBC_REWARDS": "OCBC Rewards Card",
}


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


def detect_section(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["eligible", "qualification", "who can apply"]):
        return "eligibility"
    if any(k in t for k in ["definition", "interpretation"]):
        return "definitions"
    if any(k in t for k in ["reward", "points", "miles", "cashback"]):
        return "rewards"
    if any(k in t for k in ["promotion", "bonus", "campaign"]):
        return "promotion"
    if any(k in t for k in ["exclude", "not eligible", "will not earn"]):
        return "exclusions"
    if any(k in t for k in ["annual fee", "fee", "charge"]):
        return "fees"
    if any(k in t for k in ["cap", "maximum", "limit"]):
        return "limits"
    return "general"


def get_retriever(card_keys: list[str] | None = None):
    search_kwargs: dict = {"k": 4}
    if card_keys:
        search_kwargs["filter"] = {"card_key": {"$in": card_keys}}
    return get_vector_store().as_retriever(search_kwargs=search_kwargs)


@st.cache_resource
def get_chat_model():
    return init_chat_model(os.getenv("CHAT_MODEL"))


def lookup_mcc_codes(merchant: str) -> list[dict]:
    try:
        resp = requests.get(f"https://heymax.ai/api/v2/merchant/{merchant}", timeout=5)
        resp.raise_for_status()
        return resp.json().get("data", {}).get("mcc_v2", [])
    except Exception:
        return []


def build_prompt(question: str, context: str, mcc_entries: list[dict] | None = None) -> str:
    mcc_section = ""
    if mcc_entries:
        lines = "\n".join(
            f"- MCC {e['mcc_code']}: {e['mcc_desc']} ({e['l2_category']})"
            for e in mcc_entries
        )
        mcc_section = f"\nMerchant MCC codes:\n{lines}\n"

    return f"""You are a helpful assistant that helps users maximise credit card miles and rewards in Singapore.
Answer based only on the provided context from the credit card benefit guides.
If the answer is not in the context, say you don't know.
{mcc_section}
Context:
{context}

Question:
{question}

Answer:"""


def ingest_pdf_urls(card_urls: dict[str, str]) -> tuple[int, list[str]]:
    all_chunks = []
    errors = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)

    for card_key, url in card_urls.items():
        url = url.strip()
        if not url:
            continue
        try:
            loader = PyPDFLoader(url, mode="page")
            pages = loader.load()

            # Group pages by detected section
            section_texts: dict[str, list[str]] = {}
            for page in pages:
                section = detect_section(page.page_content)
                section_texts.setdefault(section, []).append(page.page_content)

            # Merge pages within each section, then split
            card_name = CARD_ALIASES.get(card_key, card_key)
            for section, texts in section_texts.items():
                merged = Document(
                    page_content="\n\n".join(texts),
                    metadata={"card_key": card_key, "card_name": card_name, "section": section},
                )
                chunks = splitter.split_documents([merged])
                for chunk in chunks:
                    chunk.page_content = (
                        f"Card: {card_name}\nSection: {section}\n\n{chunk.page_content}"
                    )
                all_chunks.extend(chunks)
        except Exception as exc:
            errors.append(f"{CARD_ALIASES.get(card_key, card_key)}: {exc}")

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


st.set_page_config(page_title="Credit Card Miles Agent", page_icon="💳")
st.title("💳 Credit Card Miles Agent")
st.caption("Ask questions about miles and rewards for Singapore credit cards.")

with st.sidebar:
    st.header("Data Management")

    if st.button("Reload card data", use_container_width=True):
        configured = {k: v for k, v in CARD_URLS.items() if v.strip()}
        if not configured:
            st.warning("No card URLs configured in CARD_URLS.")
        else:
            with st.spinner("Clearing existing data..."):
                clear_current_collection()
            with st.spinner("Indexing card benefit guides..."):
                chunk_count, errors = ingest_pdf_urls(configured)
            st.cache_resource.clear()
            st.toast(f"Loaded {len(configured)} cards ({chunk_count} chunks).", icon="✅")
            if errors:
                st.error("Some cards failed to load:")
                for err in errors:
                    st.write(f"- {err}")

selected_aliases = st.multiselect(
    "Filter cards (leave empty to search all):",
    options=list(CARD_ALIASES.values()),
    default=list(CARD_ALIASES.values()),
)
selected_keys = [k for k, v in CARD_ALIASES.items() if v in selected_aliases]

query = st.text_input("Ask about miles and rewards:")
merchant = st.text_input("Merchant (optional — for MCC-based bonus lookup):")

if query:
    retriever = get_retriever(selected_keys if selected_keys != list(CARD_ALIASES.keys()) else None)
    model = get_chat_model()
    docs = retriever.invoke(query)

    mcc_entries = lookup_mcc_codes(merchant) if merchant.strip() else []
    if merchant.strip() and mcc_entries:
        codes = ", ".join(e["mcc_code"] for e in mcc_entries)
        st.caption(f"MCC codes for **{merchant}**: {codes}")
    elif merchant.strip():
        st.caption(f"No MCC data found for **{merchant}**.")

    if not docs:
        st.warning("No matching chunks found.")
    else:
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = build_prompt(query, context, mcc_entries)
        response = model.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("Retrieved chunks"):
            for idx, doc in enumerate(docs, start=1):
                card_name = doc.metadata.get("card_name", "unknown")
                section = doc.metadata.get("section", "unknown")
                st.markdown(f"**Chunk {idx}** — {card_name}, section `{section}`")
                st.write(doc.page_content)
                st.divider()
