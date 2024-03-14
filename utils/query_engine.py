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

from llama_index.legacy.retrievers import RecursiveRetriever
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.llms import OpenAI
from llama_index.legacy.callbacks import CallbackManager, LlamaDebugHandler
from pathlib import Path
from build_vector_db import UnstructuredVectorStore
from llama_index.legacy.tools import ToolMetadata, QueryEngineTool
from llama_index.legacy.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.legacy.agent import OpenAIAgent, FnRetrieverOpenAIAgent
from pymupdf import PymuPDF

nest_asyncio.apply()
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
os.environ["OPENAI_API_KEY"] = "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"

nltk.data.path.append('/Users/nyoosteven/Data Science/Intern/OCBC/rag_presentasi copy/nltk_data')

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

class MultiDocumentQueryEngine():

    def __init__(self,):
        self.agents_dict = {}
        self.query_engine = {}
        self.prod_qe = []
        self.unstructured = UnstructuredVectorStore()
        self.pymupdf = PymuPDF()

    def build_document_qe(self, html_folder, nodes_folder, summary_folder, file, option='html'):
        
        if option=='html':
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
            query_engine = self.build_query_engine_document(base_nodes, node_mappings)
        else:
            retriever = VectorStoreIndex(base_nodes).as_retriever(similarity_top_k=3)
            query_engine = RetrieverQueryEngine.from_args(retriever)

        qe_tool = QueryEngineTool(
                        query_engine = query_engine,
                        metadata = ToolMetadata(
                            name = f"{file_name}",
                            description = prompt+summary,
                        )
                    )
        
        return qe_tool,name
    
    def build_query_engine(self, html_folder, nodes_folder, summary_folder, option):

        for file in tqdm(os.listdir(html_folder)):
            
            qe_tool, name = self.build_document_qe(html_folder=html_folder,
                                   nodes_folder=nodes_folder,
                                   summary_folder=summary_folder,
                                   file=file,
                                   option = option)
            
            if name not in self.agents_dict.keys():
                self.agents_dict[name]=[]
            self.agents_dict[name].append(qe_tool)
        
        for name in self.agents_dict.keys():

            agent = OpenAIAgent.from_tools(
                self.agents_dict[name],
                llm=llm,
                verbose=True,
                system_prompt = f"""\
                                You are a specialized agent designed to answer queries about {name}.
                                You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
                                """
            )

            self.prod_qe.append(QueryEngineTool(query_engine=agent,
                                        metadata=ToolMetadata(
                                        name=f"tool_{name}",
                                        description=deskripsi.format(document=name),
            )))

        return self.prod_qe

    def build_query_engine_document(self, raw_nodes, node_mappings):
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
        query_engine = RetrieverQueryEngine.from_args(recursive_retriever)
        return query_engine

    def multi_documents_query_engine(self):
        tool_mapping = SimpleToolNodeMapping.from_objects(self.prod_qe)
        obj_index = ObjectIndex.from_objects(self.prod_qe,
                                             tool_mapping,
                                             VectorStoreIndex)
        
        system_prompt=""" \
        You are an agent designed to answer queries about a set of given documents.
        Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
        """
        top_agent = FnRetrieverOpenAIAgent.from_retriever(obj_index.as_retriever(similarity_top_k=3),
                                                        llm=llm,
                                                        system_prompt=system_prompt,
                                                        verbose=True)
        return top_agent


