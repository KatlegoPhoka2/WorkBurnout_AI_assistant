import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from retriever import get_retriever

load_dotenv()


def get_llm():
    llm = ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        openai_api_base="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Burnout AI Assistant"
        }
    )
    return llm


def get_prompt_template():
    template = """
You are a compassionate and knowledgeable workplace wellness assistant 
specializing in burnout prevention and recovery.

Use the context below to answer the user's question in a supportive, 
clear, and practical way. If the answer is not in the context, say 
"I don't have enough information on that, but I recommend speaking 
with an HR professional or mental health counselor."

Do NOT make up information. Always ground your response in the context provided.

Context:
{context}

User Question:
{question}

Your Response:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    return prompt


def format_docs(docs):
    """Concatenate retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    llm = get_llm()
    retriever = get_retriever(k=3)
    prompt = get_prompt_template()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain ready")
    return rag_chain


def ask(query: str):
    chain = build_rag_chain()
    answer = chain.invoke(query)

    print(f"\nQuestion: {query}")
    print(f"\nAnswer:\n{answer}")

    return answer


if __name__ == "__main__":
    ask("how to deal with work burnout?")




