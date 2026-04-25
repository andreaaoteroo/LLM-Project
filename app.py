import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IbisIQ – UM Policy Assistant",
    layout="centered"
)

# ── UM Brand CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Source+Sans+3:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

/* Background */
.stApp {
    background-color: #f7f5f2;
}

/* Header block */
.ibis-header {
    background: linear-gradient(135deg, #005030 0%, #003d24 100%);
    border-radius: 12px;
    padding: 28px 32px 20px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 4px 20px rgba(0,80,48,0.18);
}
.ibis-title {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 3rem !important;
    font-weight: 700;
    color: #f47321;
    margin: 0;
    letter-spacing: -0.5px;
}
.ibis-subtitle {
    font-size: 1.3rem;
    color: #005030;
    margin: 4px 0 0 0;
}

/* Suggestion chips */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 20px;
}
.chip {
    background: #e8f5e9;
    border: 1.5px solid #005030;
    color: #005030;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}
.chip:hover {
    background: #005030;
    color: white;
}

/* Chat messages */
.stChatMessage {
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}

/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #fff8f0 !important;
    border-left: 3px solid #F47321 !important;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #f0f7f4 !important;
    border-left: 3px solid #005030 !important;
}

/* Input box */
.stChatInput textarea {
    border: 2px solid #005030 !important;
    border-radius: 10px !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
.stChatInput textarea:focus {
    border-color: #F47321 !important;
    box-shadow: 0 0 0 2px rgba(244,115,33,0.15) !important;
}

/* Divider */
hr {
    border-color: #d4e8dd;
}

/* Footer */
.ibis-footer {
    text-align: center;
    font-size: 0.75rem;
    color: #888;
    margin-top: 32px;
    padding-top: 12px;
    border-top: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 4])
with col1:
    st.image("Miami_Hurricanes_logo.svg.png", width=110)
with col2:
    st.markdown("""
    <div style="padding-top:10px;">
        <p class="ibis-title">IbisIQ</p>
        <p class="ibis-subtitle">University of Miami · Policy Assistant</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

# ── Load API key ──────────────────────────────────────────────────────────────
api_key = st.secrets["OPENAI_API_KEY"]

# ── Load RAG pipeline (cached so it only runs once) ───────────────────────────
@st.cache_resource
def load_rag_pipeline():
    loader  = PyPDFLoader("TravelPolicy.pdf")
    loader2 = PyPDFLoader("PcardPolicy.pdf")

    pages1 = loader.load_and_split()
    pages2 = loader2.load_and_split()
    pages  = pages1 + pages2

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4.1",
        chunk_size=1000,
        chunk_overlap=50
    )
    texts = splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    db = Chroma.from_documents(texts, embeddings_model)

    client = OpenAI(api_key=api_key)

    return db, embeddings_model, client

db, embeddings_model, client = load_rag_pipeline()

# ── Helper functions ──────────────────────────────────────────────────────────
def prompt_function(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


def build_reasoning_prompt(query, chroma_results):
    documents = chroma_results["documents"][0]
    metadatas = chroma_results["metadatas"][0]

    context_items = []
    for doc, meta in zip(documents, metadatas):
        source_label = meta.get("section_id", "General Policy")
        context_items.append(f"SOURCE [{source_label}]: {doc}")

    context_str = "\n\n".join(context_items)

    system_prompt = f"""
    You are a Policy Compliance Officer. Use the provided context to answer the query.

    STRICT REQUIREMENTS:
    1. REASONING: Explain 'why' something is allowed or denied based on the rules.
    2. CITATIONS: You must cite the [Source ID] for every specific rule mentioned.
    3. NO QUOTING: Paraphrase the policy into actionable advice.
    4. UNCERTAINTY: If the source doesn't cover a specific detail, explicitly state that.

    CONTEXT FROM UNIVERSITY POLICY:
    {context_str}

    USER QUERY: {query}

    COMPLIANCE ANALYSIS:
    """
    return system_prompt


def RAG_final(query: str) -> str:
    embedding_vector = embeddings_model.embed_query(query)
    raw_results = db.similarity_search_by_vector(embedding_vector, k=10)

    transformed_results = {
        "documents": [[doc.page_content for doc in raw_results]],
        "metadatas": [[doc.metadata for doc in raw_results]]
    }

    final_prompt = build_reasoning_prompt(query, transformed_results)
    return prompt_function(final_prompt)

# ── Suggested questions ───────────────────────────────────────────────────────
st.markdown("**💡 Try asking:**")
suggestions = [
    "Can alcohol be expensed on a travel card?",
    "Are gift cards allowed on a Pcard?",
    "What receipts are required for reimbursement?",
    "What is the per diem for international travel?",
]
cols = st.columns(2)
for i, s in enumerate(suggestions):
    if cols[i % 2].button(s, key=f"suggestion_{i}", use_container_width=True):
        st.session_state["prefill"] = s

st.markdown("---")

# ── Chat interface ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I'm **IbisIQ**, your University of Miami policy assistant. Ask me anything about UM travel, expense reimbursement, or Pcard policies."
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🌴" if message["role"] == "assistant" else None):
        st.markdown(message["content"])

# Handle suggestion prefill
prefill = st.session_state.pop("prefill", None)

# Chat input
user_input = st.chat_input("Ask a policy question...") or prefill

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and show assistant response
    with st.chat_message("assistant", avatar="🌴"):
        with st.spinner("Checking policy documents..."):
            response = RAG_final(user_input)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ibis-footer">
    IbisIQ · University of Miami · Answers are based on official UM policy documents and are for guidance only.
</div>
""", unsafe_allow_html=True)
