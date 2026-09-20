"""Core RAG chain: RBAC-filtered retrieval -> guardrails -> LLM generation -> monitoring."""
import time
from config import GROQ_API_KEY, GROQ_MODEL, TOP_K, COLLECTION_NAME, CHROMA_DIR
from ingest import get_chroma_client, get_embedding_function
from rbac import build_chroma_filter
from guardrails import apply_input_guardrails, apply_output_guardrails, looks_out_of_scope
from monitoring import log_interaction

SYSTEM_PROMPT = (
    "You are an internal company assistant. Answer the user's question ONLY using "
    "the provided context extracted from company documents. If the answer is not contained in "
    "the context, say you don't have that information in the knowledge base -- do not use "
    "outside knowledge. Be concise and cite the source file names you used."
)


def retrieve(query: str, role: str, top_k: int = TOP_K):
    """Query the vector store, restricted to departments the role is authorized to see."""
    client = get_chroma_client(CHROMA_DIR)
    ef = get_embedding_function()
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
    where = build_chroma_filter(role)
    results = collection.query(query_texts=[query], n_results=top_k, where=where)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


def _get_llm():
    from langchain_groq import ChatGroq
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file (see .env.example).")
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)


def answer_query(query: str, user, role: str) -> dict:
    """Full pipeline: guardrails -> RBAC retrieval -> LLM -> guardrails -> monitoring log."""
    start = time.time()

    input_check = apply_input_guardrails(query)
    if input_check["blocked"]:
        result = {
            "answer": "This request was blocked by guardrails and cannot be processed.",
            "sources": [],
            "blocked": True,
            "reason": input_check["reason"],
        }
        log_interaction(user, role, query, result, (time.time() - start) * 1000, 0, 0)
        return result

    chunks = retrieve(query, role)

    if looks_out_of_scope(query, chunks):
        result = {
            "answer": "I don't have information about that in the company knowledge base you're "
                      "authorized to access.",
            "sources": [],
            "blocked": False,
            "reason": "out_of_scope_or_no_context",
        }
        log_interaction(user, role, query, result, (time.time() - start) * 1000, 0, 0)
        return result

    context_text = "\n\n".join(f"[Source: {m['source']}]\n{d}" for d, m in chunks)
    llm = _get_llm()
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context_text}\n\nQuestion: {query}"),
    ]
    response = llm.invoke(messages)
    raw_answer = response.content

    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)

    output_check = apply_output_guardrails(raw_answer)

    result = {
        "answer": output_check["answer"],
        "sources": sorted({m["source"] for _, m in chunks}),
        "blocked": False,
        "reason": "pii_redacted_in_output" if output_check["redacted"] else None,
    }

    log_interaction(user, role, query, result, (time.time() - start) * 1000, tokens_in, tokens_out)
    return result
