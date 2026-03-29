# Load, chunk, embed documents

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader,PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# Loading data 

def load_pdf_documents():
        
  pdf_loader = PyPDFLoader("data/documents/workbrnout.pdf")
  pdf_docs = pdf_loader.load()
  return pdf_docs


def load_web_documents():
    urls = [
        "https://www.deloitte.com/southeast-asia/en/about/people/blogs/mental-health-at-work.html",
        "https://www.who.int/news-room/fact-sheets/detail/mental-health-at-work"
    ]
    
    web_loader = WebBaseLoader(urls)
    web_docs = web_loader.load()
    
    return web_docs

pdf_docs=load_pdf_documents()
web_docs =load_web_documents()
all_docs =pdf_docs+ web_docs

print(f"Total documents: {len(all_docs)}")

  
#Split all documents into chunks

text_splitter=CharacterTextSplitter(

    chunk_size=500,
    chunk_overlap=50
)

docs=text_splitter.split_documents(all_docs)


#Create embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


