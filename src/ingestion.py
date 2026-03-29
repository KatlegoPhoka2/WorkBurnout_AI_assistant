from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf_documents():
    pdf_loader = PyPDFLoader("data/documents/workburnout.pdf")
    return pdf_loader.load()


def load_web_documents():
    urls = [
        "https://www.deloitte.com/southeast-asia/en/about/people/blogs/mental-health-at-work.html",
        "https://www.who.int/news-room/fact-sheets/detail/mental-health-at-work"
    ]
    web_loader = WebBaseLoader(urls)
    return web_loader.load()


def load_all_documents():
    pdf_docs = load_pdf_documents()
    web_docs = load_web_documents()
    all_docs = pdf_docs + web_docs
    print(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def split_documents(all_docs):
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(all_docs)
    print(f"Total chunks after splitting: {len(docs)}")
    return docs


def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


