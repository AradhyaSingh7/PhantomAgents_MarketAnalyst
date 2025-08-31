import ast
import os
import glob
import time
import numpy as np
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from google.oauth2 import service_account
import pandas as pd
import re
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.manifold import TSNE

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
db_name = "vector_db"
creds = service_account.Credentials.from_service_account_file(
    os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'your-key-if-not-using-env')
)

vector_store = None


def parse_dict_string(s):
    """
    Extracts a dictionary from a string and returns it as a Python dict.
    Handles extra characters, single/double quotes, and Python-style dicts.
    """
    # Try to extract the first {...} block
    match = re.search(r'\{.*\}', s)
    if not match:
        raise ValueError("No dictionary found in the string")

    dict_str = match.group()

    try:
        # Safely evaluate the dict string
        result = ast.literal_eval(dict_str)
        if not isinstance(result, dict):
            raise ValueError("Parsed value is not a dictionary")
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse dictionary: {e}")


def chunking(companyNoExclusive:int):
    chunks = []
    companies_path_lst = [f"data/c{(3 - len(str(i))) * "0"}{i}" for i in range(1, companyNoExclusive)]
    try:
        companies_path_lst.remove('data/c013')
    except:pass
    for company_path in companies_path_lst:
        df = pd.read_csv(company_path + "/core.csv")
        files_summary = glob.glob(company_path + "/summary/*")
        text_loader_kwargs = {'encoding': 'utf-8'}
        documents = []
        loader = DirectoryLoader(company_path + "/summary", glob="*.*", loader_cls=TextLoader,
                                 loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = os.path.basename(doc.metadata["source"].split(".")[0])
            doc.metadata['company_name'] = df.at[0, "name"]
            documents.append(doc)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = chunks + text_splitter.split_documents(documents)

    for chunk in chunks:
        match = re.search(r"##\s*(.+?)\s*##", chunk.page_content, re.DOTALL)
        if match:
            chunk.metadata['doc_content_notes'] = match.group(1)
    return chunks

def chunking_add(addlist:list):
    chunks = []
    companies_path_lst = addlist
    for company_path in companies_path_lst:
        df = pd.read_csv(company_path + "/core.csv")
        files_summary = glob.glob(company_path + "/summary/*")
        text_loader_kwargs = {'encoding': 'utf-8'}
        documents = []
        loader = DirectoryLoader(company_path + "/summary", glob="*.*", loader_cls=TextLoader,
                                 loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = os.path.basename(doc.metadata["source"].split(".")[0])
            doc.metadata['company_name'] = df.at[0, "name"]
            documents.append(doc)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = chunks + text_splitter.split_documents(documents)

    for chunk in chunks:
        match = re.search(r"##\s*(.+?)\s*##", chunk.page_content, re.DOTALL)
        if match:
            chunk.metadata['doc_content_notes'] = match.group(1)
    return chunks


def rag_setup_vectorize_data(chunks=None,new=False):
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004", location="us-central1", credentials=creds)
    if new:
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    else:
        vectorstore = Chroma(
            persist_directory=db_name,
            embedding_function=embeddings
        )
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")
    global vector_store
    vector_store = vectorstore
    return vectorstore

def llm_init(vectorstore,system_prompt="",model="gemini-2.5-flash"):
    if len(system_prompt) == 0:
        system_prompt = """
You are a predictive market research analyst. Your primary function is to generate data-driven forecasts for future years by extrapolating from the information provided in the context.

Core Directives:
Analyze the User's Intent: First, determine if the query is asking for a hard fact, a forecast, or general knowledge.

Formulate Your Response (Using Bullet Points):

For Factual Questions (e.g., "What were the sales figures for Q2?"):

Provide a direct, one-sentence answer based strictly on the context.

If the information is not available, state: "The provided context does not contain this specific information."

For Predictive & Forecasting Questions (e.g., "Project Helios's market share for the next three years."):

This is your primary function. Your goal is to extrapolate from the provided data to generate specific, quantitative forecasts.

First, identify key performance indicators (KPIs) and historical trends from the context (e.g., year-over-year growth, market share changes).

Use these data points as a baseline to project future figures and trends. You must generate specific numbers and percentages.

Present your forecast as a series of clear, concise bullet points. Each point should represent a specific prediction.

Example Output:

Projected revenue growth for 2026: 12-15%

Anticipated increase in market share by 2027: ~5%

Predicted reduction in operational costs: -8%

For General Questions (e.g., "What is a SWOT analysis?"):

Provide a brief, clear explanation from your general knowledge base in a bullet-point format.
        """
    llm = ChatVertexAI(model_name=model, location="us-central1")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {input}\n\nContext:\n{context}")
    ])
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    return create_retrieval_chain(retriever, doc_chain)

def llm_response(message,llm):
    result_in = llm.invoke({"input": message})
    return result_in["answer"]

def init_chain(new=False,add=False,adding_list=None):
    if new:
        chunks = chunking(6)
        vectorstore = rag_setup_vectorize_data(chunks,True)
    else:
        vectorstore = rag_setup_vectorize_data()
    if add:
        for list_ in adding_list:
            chunks = chunking_add(list_)
            vectorstore.add_documents(chunks)
            time.sleep(70)
    return llm_init(vectorstore)

def graph_plot_llm(query):
    system = """
            You are an AI business classifier.
        Your only task is to classify the user’s query into one of the following categories:
        
        ["core", "documentation", "finances", "market", "stockdata", "macro_regulatory_data", "sentiment_external_opinions", "general","events_interactions"]
        
        Rules:
        
        Always return the result as a valid JSON string in the format:
        
        {"result": "<one_of_the_categories>"}
        
        
        If the query does not clearly fit into any of the categories, classify it as "general".
        
        Never add extra commentary, reasoning, or text outside of the JSON.
        
        Be strict: pick the closest matching category based on intent.
        
        Examples:
        
        User Query: "Show me the quarterly revenue breakdown of Tesla."
        Response:
        
        {"result": "finances"}
        
        
        User Query: "What are analysts saying about Apple’s new iPhone release?"
        Response:
        
        {"result": "sentiment_external_opinions"}
        
        
        User Query: "Plot the last 30 days of Amazon’s stock prices."
        Response:
        
        {"result": "stockdata"}
        
        
        User Query: "Summarize the new EU AI regulatory proposal."
        Response:
        
        {"result": "macro_regulations"}
        
        
        User Query: "Upload the compliance PDF report for Microsoft."
        Response:
        
        {"result": "documentation"}
        
        
        User Query: "How is the overall retail market performing this year?"
        Response:
        
        {"result": "market"}
        
        
        User Query: "What is the mission and core business of Netflix?"
        Response:
        
        {"result": "core"}
        
        
        User Query: "Hey, what’s up?"
        Response:
        
        {"result": "general"}
    """
    llm = ChatVertexAI(model_name="gemini-2.5-flash-lite", location="us-central1")
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=query)
    ]
    print(llm.invoke(messages).content)
    print(" parse string :  ",parse_dict_string(llm.invoke(messages).content))
    response = parse_dict_string(llm.invoke(messages).content)
    catagory = response["result"]

    collection = vector_store._collection
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])

    vectors = np.array(result['embeddings'])
    data_type = [meta['doc_type'] for meta in result['metadatas']]
    if catagory == "general":
        tsne = TSNE(n_components=2, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)
        extracted_data = [[float(point) for point in list(item)] for item in list(reduced_vectors) ]
        return (extracted_data,catagory)
    else:


        sorted_list_zipped = list(filter( lambda obj: obj[1] == catagory ,zip(list(vectors),data_type)))
        for item in zip(list(vectors),data_type):
            if item[1] == catagory:
                print("matched ")
            else:
                print("not matched  :  ",catagory,"   :    ",item[1])

        sorted_list_vectors = np.array([item[0] for item in sorted_list_zipped])
        tsne = TSNE(n_components=2, random_state=42)
        reduced_vectors = tsne.fit_transform(sorted_list_vectors)

        extracted_data = [[float(point) for point in list(item)] for item in list(reduced_vectors)]
        return (extracted_data, catagory)

#init_chain(False,True,[[ 'data/c017', 'data/c018', 'data/c019', 'data/c020', 'data/c021']])




