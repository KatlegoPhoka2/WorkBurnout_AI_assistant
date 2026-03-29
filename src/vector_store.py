from langchain_community.vectorstores import FAISS

# FAISS index: build, save, load
# Step-4 Store document chunks into FAISS vector database

vectorstore=FAISS.from_documents(docs,embeddings)

# Step-5 Create retriever

#retriever=vectorstore.as_retriever(search_kwargs={"k":3})


from langchain_community.vectorstores import FAISS
from ingest import load_all_documents, split_documents, get_embeddings


def build_vectorstore():
    all_docs = load_all_documents()
    docs = split_documents(all_docs)
    embeddings = get_embeddings()

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    print("FAISS index saved to faiss_index/")

    return vectorstore


def load_vectorstore():
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded from faiss_index/")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()