"""GenAI Chat With Your PDF - Streamlit web app."""
import streamlit as st

from utils.pdf_indexer import index_pdf, drop_collection, get_embedder
from utils.rag_chain import ask

st.set_page_config(page_title="Chat With Your PDF", page_icon="📄", layout="wide")

# ---- session state (survives Streamlit's page re-runs) ----
defaults = {"messages": [], "collection": None, "file_key": None, "info": ""}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def load_model():
    return get_embedder()


# ---- sidebar: upload ----
with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded is not None:
        file_key = f"{uploaded.name}-{uploaded.size}"
        if file_key != st.session_state.file_key:
            # new file: process it and start a fresh chat
            old = st.session_state.collection
            old_name = old.name if old is not None else None
            st.session_state.messages = []
            st.session_state.collection = None
            st.session_state.info = ""
            try:
                with st.spinner("Loading model (first time only) ..."):
                    load_model()
                with st.spinner("Reading and indexing the PDF ..."):
                    collection, n_pages, n_chunks = index_pdf(uploaded, old_name)
                st.session_state.collection = collection
                st.session_state.file_key = file_key
                st.session_state.info = f"{uploaded.name}: {n_pages} pages, {n_chunks} chunks"
            except ValueError as e:
                st.session_state.file_key = None
                st.error(str(e))
            except Exception as e:
                st.session_state.file_key = None
                st.error(f"Something went wrong while processing the PDF: {e}")
    else:
        # file removed with the X button: free its memory
        if st.session_state.collection is not None:
            drop_collection(st.session_state.collection.name)
        st.session_state.collection = None
        st.session_state.file_key = None
        st.session_state.messages = []
        st.session_state.info = ""

    if st.session_state.collection is not None:
        st.success("Processed and ready for questions")
        st.caption(st.session_state.info)
    st.caption("Answers are generated only from the uploaded PDF.")

# ---- main: chat ----
st.title("Chat With Your PDF")
st.write("Upload a document and ask questions about it in plain language.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("pages"):
            with st.expander("Sources: pages " + ", ".join(map(str, msg["pages"]))):
                for page, text in msg["chunks"]:
                    st.markdown(f"**Page {page}**")
                    st.caption(text)

if st.session_state.collection is None:
    st.info("Upload a PDF in the sidebar to start chatting.")

question = st.chat_input(
    "Ask a question about your PDF...",
    disabled=st.session_state.collection is None,
)

if question and question.strip():
    question = question.strip()
    history = list(st.session_state.messages)  # messages before this question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking ..."):
                answer, pages, chunks = ask(question, st.session_state.collection, history)
            st.write(answer)
            if pages:
                with st.expander("Sources: pages " + ", ".join(map(str, pages))):
                    for page, text in chunks:
                        st.markdown(f"**Page {page}**")
                        st.caption(text)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "pages": pages, "chunks": chunks}
            )
        except Exception as e:
            st.error(str(e))
            # remove the unanswered question so the chat stays consistent
            st.session_state.messages.pop()
