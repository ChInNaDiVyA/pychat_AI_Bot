import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.callbacks import FileCallbackHandler
from loguru import logger
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv('/Users/abhinavnagaboyina/Documents/Gen_AI_chat_bot/.env')

llm = OpenAI(openai_api_key= os.getenv("OPENAI_API_KEY"))

st.set_page_config(
        page_title="A Friendly Bot",
        page_icon="🤖 🗣️"
    )


if "messages" not in st.session_state:
    st.session_state.messages= []
    
st.header("A Friendly Bot 🤖🗣️ ")

user_input = st.sidebar.text_input("Enter your query and dont forget to hit enter:")

logfile = "output.log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)

model = ChatOpenAI(streaming=True,callbacks=[StreamingStdOutCallbackHandler()],temperature=0.7)

sys_template= "You are a helpful assistant that assists users, user query is {query}"
prompt = PromptTemplate(
   input_variables=["query"],
   template=sys_template
)

chain1 = LLMChain(llm=model, prompt=prompt, callbacks=[handler], verbose=True)


if user_input:
    st.session_state.messages.append(HumanMessage(content= user_input))

    with get_openai_callback() as cb:
        response= chain1.run(user_input)
    st.session_state.messages.append(AIMessage(content=response))

    total_tokens = cb.total_tokens
    st.sidebar.write(f"Total Tokens: {cb.total_tokens}")
    st.sidebar.write(f"Total Cost (USD): ${cb.total_cost}")

messages = st.session_state.messages       

for i, msg in enumerate(messages):
    if i%2==0:
        message(msg.content, is_user= True)
    else:
        message(msg.content, is_user= False)

