import torch
import chromadb
import time
import os
import cProfile
import pstats
from pstats import SortKey

from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.extractors import TitleExtractor
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.evaluation import (FaithfulnessEvaluator,
                                         QueryResponseEvaluator,
                                         DatasetGenerator,
                                         RelevancyEvaluator)
import asyncio
import nest_asyncio
nest_asyncio.apply()


# setting for GPU environment
def setting_for_env():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4000"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("finish setting for env\n")


# load for the uploaded files
def data_loading():
    reader = SimpleDirectoryReader(input_dir="data/", recursive=True)
    documents = reader.load_data(num_workers=10)
    return documents


async def parallel_ingestion(pipeline, documents):
    loop = asyncio.get_event_loop()
    nodes = await loop.run_until_complete(pipeline.arun(documents=documents, num_workers=4))
    return nodes


# RAG
def building_reg_pipeline():
    # LLM, Llama2
    Settings.llm = Ollama(model="llama2",
                          request_timeout=300.0,
                          device_map="cuda")

    # load documents, parallel programming
    # use cProfile to evaluate performance
    profiler = cProfile.Profile()
    profiler.enable()
    documents = data_loading()
    profiler.disable()
    profiler.dump_stats("newstas")

    # node parser
    # node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
    # nodes = node_parser.get_nodes_from_documents(documents)

    # chroma database initialize
    db = chromadb.PersistentClient(path="./chroma_db")

    # create collection
    chroma_collection = db.get_or_create_collection("chroma_collection")

    # vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Transformation
    start_time = time.time()
    # Settings.text_splitter = SentenceSplitter(chunk_size=1024)
    # index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=1024)],
    #                                        storage_context=storage_context, embed_model=Settings.llm)

    # index = VectorStoreIndex.from_documents(documents,
    #                                         nodes=nodes,
    #                                         storage_context=storage_context)

    # parallelizing ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=20),
            TitleExtractor(),
            OllamaEmbedding(model_name="llama2")
        ]
    )
    pipeline.disable_cache = True

    # nodes = parallel_ingestion(pipeline, documents)
    nodes = pipeline.run(documents=documents, num_works=4)
    Settings.embed_model = OllamaEmbedding(model_name="llama2")
    index = VectorStoreIndex(nodes=nodes,
                             vector_store=vector_store,
                             storage_context=storage_context)

    end_time = time.time()
    print("finish transformation, time cost: ", end_time - start_time, "s\n")

    # create query engin
    # Reorder the searched Node
    # solve the missing content problem
    start_time = time.time()
    reorder = LongContextReorder()
    query_engine = index.as_query_engine(llm=Settings.llm,
                                         node_postprocessor=[reorder],
                                         similarity_top_k=5,
                                         streaming=True)
    end_time = time.time()
    print("finish query engin creation, time cost: ", end_time - start_time, "s\n")

    return query_engine


# query from the LLM via query engine
def querying(prompt, query_engine):
    start_time = time.time()
    response = query_engine.query(prompt)
    query_time = time.time() - start_time
    print("Total query time : ", query_time, "\n")
    return response


# load for prompts(our questions)
def prompt_loading():
    with open("prompt/prompt.txt", "r") as prompt_file:
        lines = prompt_file.readlines()
        for line in lines:
            yield line.rstrip()


# load for auto generation prompts(questions), not finished yet
def auto_generator_prompt(documents):
    data_generator = DatasetGenerator.from_documents(documents)
    eval_questions = data_generator.generate_questions_from_nodes()
    yield eval_questions


def main():
    # GPU
    setting_for_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("current device: ", device)
    print("device name: ", torch.cuda.get_device_name(device), "\n")

    # load for query_engine
    query_engine = building_reg_pipeline()
    # performance evaluation of data loading
    p = pstats.Stats("newstas")
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(15)

    # load for prompt(questions)
    prompt_generator = prompt_loading()

    # querying
    for prompt in prompt_generator:
        print("Ask : ", prompt)
        response = querying(prompt, query_engine)
        print("Answer : ")
        print(str(response))


if __name__ == '__main__':
    main()
