import os
import sys
import json
import copy
import openai
import pickle
import nltk
import nest_asyncio
from tqdm import tqdm
from llama_index.legacy import (VectorStoreIndex,
                         ServiceContext,
                         set_global_service_context)

from llama_index.legacy.retrievers import RecursiveRetriever, RouterRetriever
from llama_index.legacy.llms import OpenAI
from llama_index.legacy.callbacks import CallbackManager, LlamaDebugHandler
from pathlib import Path
from build_vector_db import UnstructuredVectorStore
from llama_index.legacy.tools import RetrieverTool
from llama_index.legacy.selectors import PydanticMultiSelector
from pymupdf import PymuPDF

nest_asyncio.apply()
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
os.environ["OPENAI_API_KEY"] = "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"

nltk.data.path.append('/Users/nyoosteven/Documents/Data Science/Intern/OCBC/rag/nltk_data')

llm = OpenAI(temperature=0, 
             model="gpt-3.5-turbo",
             callback_manager= callback_manager,
             max_token = 1000)

num_output = 256
chunk_size_limit = 1000
service_context = ServiceContext.from_defaults(num_output=num_output, 
                                               chunk_size_limit=chunk_size_limit,
                                               llm=llm)

set_global_service_context(service_context)

prompt_type = {
    'prospektus':"This is the prospectus document {file_name} that investors must carefully review, comprehend, and contemplate before making investments in mutual funds. A prospectus serves as a valuable tool for investors to identify the Fund Manager and the mutual funds that will serve as their investment targets.",
    'fundsheet':"This is the fundsheets document {file_name} that issued monthly by the Fund Managers. It furnishes details regarding product performance, asset composition, and the securities portfolio at the end of each month for each mutual fund."
}

deskripsi = "This content contains about prospectus and fund sheet about {document}. Use this tool if you want to answer any questions about {document}.\n"

class MultiDocumentRetriever():

    def __init__(self,):
        self.agents_dict = {}
        self.query_engine = {}
        self.prod_retriever = []
        self.unstructured = UnstructuredVectorStore()
        self.pymupdf = PymuPDF()

    def build_document_retriever(self, html_folder, nodes_folder, summary_folder, file, option='html'):

        if option == 'html':
            file_name = str(file.split('.html')[0]).lower()
        else:
            file_name = str(file.split('.pdf')[0]).lower()

        if file_name.endswith('prospektus'):
            name = file_name.split('_prospektus')[0]
            tipe = 'prospektus'
        else:
            name = file_name.split('_fundsheet')[0]
            tipe = 'fundsheet'

        html_path = f'{html_folder}/{file_name}.{option}'
        nodes_path = f'{nodes_folder}/{file_name}.pkl'
        summary_path = f'{summary_folder}/{file_name}.txt'

        if option=='html':
            base_nodes, node_mappings, summary = self.unstructured.get_nodes_from_documents(html_path, file_name, nodes_path, summary_path)
        else:
            base_nodes, summary = self.pymupdf.get_nodes_from_documents(html_path, file_name, nodes_path, summary_path)

        prompt = prompt_type[tipe].format(file_name=name)

        if option=='html':
            retriever = self.build_recursive_retriever_document(base_nodes, node_mappings)
        else:
            retriever = VectorStoreIndex(base_nodes).as_retriever(similarity_top_k=3)

        retriever_tool = RetrieverTool.from_defaults(retriever = retriever, 
                                                description = prompt+summary)
        return retriever_tool,name
    
    def build_retriever(self, html_folder, nodes_folder, summary_folder, option):

        for file in tqdm(os.listdir(html_folder)):
            
            retriever_tool, name = self.build_document_retriever(html_folder=html_folder,
                                   nodes_folder=nodes_folder,
                                   summary_folder=summary_folder,
                                   file=file, option=option)
            
            if name not in self.agents_dict.keys():
                self.agents_dict[name]=[]
            self.agents_dict[name].append(retriever_tool)
        
        for name in self.agents_dict.keys():

            retriever = RouterRetriever(selector=PydanticMultiSelector.from_defaults(llm=llm),
                                        retriever_tools=self.agents_dict[name],)
            
            retriever = RetrieverTool.from_defaults(retriever = retriever, 
                                                    description = deskripsi.format(document=name))

            
            self.prod_retriever.append(retriever)
        
        top_agent = RouterRetriever(selector=PydanticMultiSelector.from_defaults(llm=llm),
                        retriever_tools=self.prod_retriever)
        
        return top_agent
    
    def build_recursive_retriever_document(self, raw_nodes, node_mappings):
        """
        Build retriever Query Engine
        """
        vector_index = VectorStoreIndex(raw_nodes)
        vector_retriever = vector_index.as_retriever(similarity_top_k=3)
        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict=node_mappings,
            verbose=True,
        )
        return recursive_retriever