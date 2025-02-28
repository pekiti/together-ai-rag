import os
import asyncio
import csv
import time
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.together import TogetherLLM
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.core.evaluation import RetrieverEvaluator, RelevancyEvaluator, FaithfulnessEvaluator, BatchEvalRunner

# Initialize logging as early as possible
# If using Abseil logging, ensure absl::InitializeLog() is called here

# Load environment variables from a .env file
load_dotenv()

# Ensure the TOGETHER_API_KEY environment variable is set
if "TOGETHER_API_KEY" not in os.environ:
    raise EnvironmentError("TOGETHER_API_KEY not set in environment variables")

# Initialize the vector store
vector_store_name = "mini-llama-articles"
chroma_client = chromadb.PersistentClient(path=vector_store_name)
chroma_collection = chroma_client.get_or_create_collection(vector_store_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Load and process documents from a CSV file
rows = []
try:
    with open("./mini-llama-articles.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        for idx, row in enumerate(csv_reader):
            if idx == 0: continue  # Skip header row
            rows.append(row)
except FileNotFoundError:
    raise FileNotFoundError("The file mini-llama-articles.csv was not found.")

# Convert CSV rows to Document objects
documents = [
    Document(text=row[1], metadata={"title": row[0], "url": row[2], "source_name": row[3]})
    for row in rows
]

# Define a text splitter for document segmentation
text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)

# Create an ingestion pipeline to process and store documents
pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    ],
    vector_store=vector_store
)

# Run the pipeline to process documents
nodes = pipeline.run(documents=documents, show_progress=True)

# Initialize the LLM for query processing
llm = TogetherLLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    api_key=os.environ["TOGETHER_API_KEY"]
)

# Create an index from the vector store
index = VectorStoreIndex.from_vector_store(vector_store, embed_model="local:BAAI/bge-small-en-v1.5")

# Define a query engine using the index and LLM
query_engine = index.as_query_engine(llm=llm)

# Test the query engine with a sample query
res = query_engine.query("How many parameters LLaMA 2 has?")
print(res.response)

# Print source nodes for the query result
for src in res.source_nodes:
    print("Node ID\t", src.node_id)
    print("Title\t", src.metadata['title'])
    print("Text\t", src.text)
    print("Score\t", src.score)
    print("-_"*20)

# Generate question-context pairs for evaluation
llm = OpenAI(model="gpt-4o-mini")
rag_eval_dataset = generate_question_context_pairs(
    nodes,
    llm=llm,
    num_questions_per_chunk=1
)

# Save the evaluation dataset to a JSON file
rag_eval_dataset.save_json("./rag_eval_dataset.json")

# Define an asynchronous function to run evaluations
async def run_evaluation(index, rag_eval_dataset, top_k_values, llm_judge, llm, n_queries_to_evaluate=20, num_work=1):
    evaluation_results = {}

    # Evaluate MRR and Hit Rate for different top_k values
    for top_k in top_k_values:
        retriever = index.as_retriever(similarity_top_k=top_k)
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=retriever
        )
        eval_results = await retriever_evaluator.aevaluate_dataset(rag_eval_dataset)
        avg_mrr = sum(res.metric_vals_dict["mrr"] for res in eval_results) / len(eval_results)
        avg_hit_rate = sum(res.metric_vals_dict["hit_rate"] for res in eval_results) / len(eval_results)
        evaluation_results[f"mrr_@_{top_k}"] = avg_mrr
        evaluation_results[f"hit_rate_@_{top_k}"] = avg_hit_rate

    # Evaluate faithfulness and relevancy
    queries = list(rag_eval_dataset.queries.values())
    batch_eval_queries = queries[:n_queries_to_evaluate]
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm_judge)
    relevancy_evaluator = RelevancyEvaluator(llm=llm_judge)
    runner = BatchEvalRunner(
        {
            "faithfulness": faithfulness_evaluator,
            "relevancy": relevancy_evaluator
        },
        workers=num_work,
        show_progress=True,
    )
    eval_results = await runner.aevaluate_queries(query_engine, queries=batch_eval_queries)
    faithfulness_score = sum(result.passing for result in eval_results['faithfulness']) / len(eval_results['faithfulness'])
    relevancy_score = sum(result.passing for result in eval_results['relevancy']) / len(eval_results['relevancy'])
    evaluation_results["faithfulness"] = faithfulness_score
    evaluation_results["relevancy"] = relevancy_score

    return evaluation_results

# Main asynchronous function to run evaluations and measure performance
async def main():
    top_k_values = [2, 4, 6, 8, 10]
    llm_judge = OpenAI(temperature=0, model="gpt-4o")

    llm = TogetherLLM(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        api_key=os.environ["TOGETHER_API_KEY"]
    )
    # Run evaluation with Meta-Llama-3.1-70B-Instruct-Turbo
    evaluation_results = await run_evaluation(index, rag_eval_dataset, top_k_values, llm_judge, llm=llm, n_queries_to_evaluate=20, num_work=1)
    print(evaluation_results)

    # Run evaluation with GPT-4o-mini
    llm = OpenAI(model="gpt-4o-mini")
    evaluation_results = await run_evaluation(index, rag_eval_dataset, top_k_values, llm_judge, llm=llm, n_queries_to_evaluate=20, num_work=16)
    print(evaluation_results)

    # Run evaluation with Gemini
    llm = Gemini(model="models/gemini-1.5-flash")
    evaluation_results = await run_evaluation(index, rag_eval_dataset, top_k_values, llm_judge, llm=llm, n_queries_to_evaluate=20, num_work=1)
    print(evaluation_results)

    # Measure completion time for different models
    for model_name, model_class in [("Llama 3.1 70B", TogetherLLM), ("GPT 4o Mini", OpenAI), ("Gemini 1.5 Flash", Gemini)]:
        llm = model_class(model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" if model_name == "Llama 3.1 70B" else "gpt-4o-mini" if model_name == "GPT 4o Mini" else "models/gemini-1.5-flash")
        time_start = time.time()
        llm.complete("List the 50 states in the United States of America. Write their names in a comma-separated list and nothing else.")
        time_end = time.time()
        print(f"Time taken for {model_name}: {time_end - time_start:.2f} seconds")

# Run the main function
asyncio.run(main())