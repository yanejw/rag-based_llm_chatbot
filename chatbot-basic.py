"""
Test code of simple question and answer bot.
"""

from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage

# instantiate the llm 
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat_model = ChatHuggingFace(llm=llm)

# ask the question
messages = [
    SystemMessage(content="You're a helpful research assistant"),
    HumanMessage(content="Why is Easter celebrated?")
]

ai_msg = chat_model.invoke(messages)

print(f"\n{ai_msg.content} \n")
