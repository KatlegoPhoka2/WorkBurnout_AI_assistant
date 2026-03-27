# Load, chunk, embed documents

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader,PyPDFDirectoryLoader

def load_pdf_documents():
        
  pdf_loader = PyPDFDirectoryLoader("path")
  pdf_docs = pdf_loader.load()
  return pdf_docs


def load_web_documents():
  
  web_loader = WebBaseLoader("https://example.com")
  web_docs = web_loader.load()
  return web_docs

pdf_docs = load_pdf_documents("data/pdfs/")

web_docs = load_web_documents([
    "https://example.com",
    "https://another-site.com"
])

all_docs = pdf_docs + web_docs

print(f"Total documents: {len(all_docs)}")

  



