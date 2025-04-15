import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_astradb import AstraDBVectorStore
from langchain_core.runnables import Runnable

load_dotenv()

# Load from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Build LangChain components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = AstraDBVectorStore(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT
)

custom_prompt = PromptTemplate(
    input_variables=["context", "question", "device"],
    template="""
You are a helpful and knowledgeable assistant trained to answer user questions about the medical device: {device}.

Use only the provided context from retrieved relevant documents. If the exact answer is not explicitly stated, infer the most likely answer based on related information within the retrieved documents. If the retrieved documents do not contain sufficient information, provide a **very brief**, cautious response based on what is generally true for this type of device — but do **not** fabricate or guess specific details.

Avoid referencing other devices or external knowledge not present in the documents.

Always be clear, accurate, and specific to the {device}.

Context:
{context}

Question:
{question}

Answer:
"""
)


try:
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
except Exception as e:
    raise RuntimeError(f"Failed to create retriever from vector store: {str(e)}")

try:
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",
        temperature=0
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChatOpenAI model: {str(e)}")

try:
    qa_chain: Runnable = custom_prompt | llm
except Exception as e:
    raise RuntimeError(f"Failed to compose LLM chain from prompt and model: {str(e)}")


def run_query(query: str, device: str):
    try:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            raise ValueError("No documents were retrieved for the given query.")

        try:
            context = "\n\n".join(doc.page_content for doc in docs)
        except Exception as format_err:
            raise ValueError(f"Failed to format context: {format_err}")
        
        try:
            answer = qa_chain.invoke({
                "context": context,
                "question": query,
                "device": device
            })
        except Exception as llm_err:
            raise RuntimeError(f"LLM invocation failed: {llm_err}")

        sources = [doc.page_content[:500] for doc in docs]
        return answer, sources

    except Exception as e:
        # Log or re-raise depending on how you handle it
        raise RuntimeError(f"run_query() failed: {str(e)}")
