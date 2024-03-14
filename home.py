import streamlit as st 
import sys
import os
sys.path.append('/Users/nyoosteven/Data Science/Intern/OCBC/rag_presentasi/utils')
from utils.build_vector_db import UnstructuredVectorStore
from utils.query_engine import MultiDocumentQueryEngine

st.set_page_config(
    page_title="Your RAGs Searchbot, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Your RAGs Searchbot, powered by LlamaIndex 💬🦙")

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        multiDocumentEngine = MultiDocumentQueryEngine()
        multiDocumentEngine.build_query_engine(html_folder = '/Users/nyoosteven/Data Science/Intern/OCBC/rag_presentasi copy/data/pdf',
                                       nodes_folder = '/Users/nyoosteven/Data Science/Intern/OCBC/rag_presentasi copy/data/nodes_pdf',
                                       summary_folder = '/Users/nyoosteven/Data Science/Intern/OCBC/rag_presentasi copy/data/summary_pdf',
                                       option='pdf')
        top_agent_query_engine = multiDocumentEngine.multi_documents_query_engine()
        return top_agent_query_engine

top_agent = load_data()

search_term = st.text_input(
            label=":blue[Search your Query]",
            placeholder="Your query...",)

search_button = st.button(label="Run", type="primary")

if search_term:
    # And if they have clicked the search button
    if search_button:
        response = top_agent.query(search_term)
        st.write(response.response)
        #st.write(response.response)
        for text in response.source_nodes:
            output = text.text
            output = output.replace('\n',' ').split()
            st.text_area(label=text.node_id,value=' '.join(output[:50])+'...')