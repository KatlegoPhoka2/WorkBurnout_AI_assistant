from vector_store import load_vectorstore


def get_retriever(k=3):
    """
    Load the FAISS vectorstore and return a retriever.
    k = number of relevant chunks to retrieve per query.
    """
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    print(f"Retriever ready — returning top {k} chunks per query")
    return retriever


def retrieve_docs(query: str, k=3):
    """
    Quick helper to retrieve relevant chunks for a given query string.
    Useful for testing retrieval quality before hooking up the LLM.
    """
    retriever = get_retriever(k=k)
    results = retriever.invoke(query)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Chunk {i} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Content: {doc.page_content[:300]}...")
        print()

    return results


if __name__ == "__main__":
    retrieve_docs("What are the signs of workplace burnout?")
