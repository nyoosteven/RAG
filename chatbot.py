import streamlit as st 
import sys
import os
sys.path.append('/Users/nyoosteven/Data Science/Intern/OCBC/rag_presentasi/utils')
from utils.build_vector_db import UnstructuredVectorStore
from utils.query_engine import MultiDocumentQueryEngine

st.set_page_config(
    page_title="Your RAGs Chatbot, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Your RAGs Chatbot, powered by LlamaIndex 💬🦙")

# with st.sidebar:
#     st.title('Welcome Abroad 🦙')
#     openai_token = st.text_input('Open AI token: ')

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Greetings, I'm here to assist you"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        multiDocumentEngine = MultiDocumentQueryEngine()
        multiDocumentEngine.build_query_engine(html_folder = '/Users/nyoosteven/Data Science/Intern/OCBC/rag_presentasi/data/pdf',
                                       nodes_folder = '/Users/nyoosteven/Data Science/Intern/OCBC/rag_presentasi/data/nodes_pdf',
                                       summary_folder = '/Users/nyoosteven/Data Science/Intern/OCBC/rag_presentasi/data/summary_pdf',
                                       option='pdf')
        top_agent_query_engine = multiDocumentEngine.multi_documents_query_engine()
        return top_agent_query_engine

top_agent = load_data()

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = top_agent.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)