import os
import sys
import json
import copy
import openai
import pickle
import nltk 
import nest_asyncio
from typing import Optional
 
from llama_index.legacy import (VectorStoreIndex,
                         ServiceContext,
                         StorageContext,
                         SimpleDirectoryReader,
                         SummaryIndex,
                         set_global_service_context)

from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.llms import OpenAI
from llama_index.legacy.node_parser import UnstructuredElementNodeParser
from llama_index.core.schema import IndexNode
from llama_index.legacy.retrievers import RecursiveRetriever
from llama_index.legacy.readers.file.flat_reader import FlatReader
from llama_index.legacy.callbacks import CallbackManager, LlamaDebugHandler
from pathlib import Path

nest_asyncio.apply()
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
os.environ["OPENAI_API_KEY"] = "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"

nltk.data.path.append('/Users/nyoosteven/Data Science/Intern/OCBC/rag_presentasi/nltk_data')

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

def write_text(path, text):
    with open(path, "w+") as file:
        file.write(text)

def load_text(path):
    with open(path, "r") as file:
        text = file.read()
    return text

class UnstructuredVectorStore():

    def __init__(self):
        # inisialisi
        self.reader = FlatReader()
        self.node_parser = UnstructuredElementNodeParser()

    def get_nodes_from_documents(self, file_name, product_name, node_save_path, summary_path):

        tabel, now = 1,1
        node_tmp_mapping={}
        if 'html' in file_name:
            docs = self.reader.load_data(Path(file_name))
            if node_save_path is None or not os.path.exists(node_save_path):

                raw_nodes = self.node_parser.get_nodes_from_documents(docs, True)
                summary = self.get_summary(raw_nodes)
                pickle.dump(raw_nodes, open(node_save_path, 'wb'))
                write_text(summary_path, summary)

            else:

                raw_nodes = pickle.load(open(node_save_path, 'rb'))
                summary = load_text(summary_path)

            for node in raw_nodes:
                node.metadata.update({'file_name':product_name})
            
            for node in raw_nodes:
                if node.node_id.endswith('ref'):
                    node_tmp_mapping[node.node_id]=f"{product_name}-node-table-ref-{tabel}"
                elif node.node_id.endswith('table'):
                    node_tmp_mapping[node.node_id]=f"{product_name}-node-table-{tabel}"
                    tabel+=1
                else:
                    node_tmp_mapping[node.node_id]=f"{product_name}-node-{now}"
                    now+=1
            
            for node in raw_nodes:
                node.id_ = node_tmp_mapping[node.id_]
                for relationship in node.relationships.values():
                    try:
                        relationship.node_id = node_tmp_mapping[relationship.node_id]
                    except:
                        pass
                if str(type(node))=="<class 'llama_index.legacy.schema.IndexNode'>":
                    node.index_id = node_tmp_mapping[node.index_id]
            
            base_nodes, node_mappings = self.node_parser.get_base_nodes_and_mappings(raw_nodes)
            return base_nodes, node_mappings, summary
        else:
            print("only parsing pdf")
            return 
    

    def get_summary(self,raw_nodes):

        summary_index = SummaryIndex(raw_nodes) 
        summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", llm=llm)
        summary = str(summary_query_engine.query("Extract a concise 1-2 line summary of this document"))
        return summary
        

