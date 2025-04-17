"""
Test code using a direct question and answer from Hugging Face chat models and embeddings.
"""

from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

from langchain import hub
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

################################################################################################
################################################################################################
################################################################################################

# instantiate the llm 
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", ## repo_id refers to the repo on hugging face | this one here is a llm model
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)
chat_model = ChatHuggingFace(llm=llm)

################################################################################################
################################################################################################
################################################################################################

# get the embeddings
# necessary for rag because it needs to embed from the knowledge base
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

################################################################################################
################################################################################################
################################################################################################

# store embeddings in a vector db
# so that if your document changes, the embeddings are still in the db
vector_store = InMemoryVectorStore(embeddings)

################################################################################################
################################################################################################
################################################################################################

# load the knowledge base
loader = UnstructuredWordDocumentLoader(
    "gmv_calculations.docx",
    mode="elements",
    strategy="fast",
)
docs = loader.load()

# split text into chunks as llm is unable to take in everything at once
# the smaller the chunks, the more accurate, but will take more compute
# chunk overlap gives context to the chunks (before and after)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30, add_start_index=True)
all_splits = text_splitter.split_documents(docs)
document_ids = vector_store.add_documents(documents=all_splits)

################################################################################################
################################################################################################
################################################################################################

# start q and a session
prompt = hub.pull("rlm/rag-prompt")

# define state for app
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = chat_model.invoke(messages)
    return {"answer": response.content}

# compile application
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "How is Tiktok gmv calculated?"})
print(response["answer"])
