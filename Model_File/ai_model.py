from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Dict
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
import pdfplumber
import json
import os 
import fitz
import pytesseract
import string
import shutil




#Loading .env file
load_dotenv()

# Groq related Global variables
groq_api_key=os.getenv("Groq_API_Key")
groq_temperature=os.getenv("Groq_Temperature")
groq_model_name=os.getenv("Groq_Model_Name")

# Ollama related Global variables
ollama_embedding_model_name=os.getenv("Ollama_Embedding_Model_Name")
ollama_model_temperature=os.getenv("Ollama_Embedding_Model_Temperature")


# Pydantic Class
class LLM_Structured_Output(BaseModel):
    answer:str=Field("This key contain the answer provided by the llm based on the provided context.")
    confidence:int=Field("This key contain how much confidence llm have on the provided output.")
    risk_level:Literal["Low","Medium","High"]=Field("This key contain the risk in the provided answer for taking decision.")
    decision:Literal["Show answer","Show Warning","Block Answer","Ask user to rephrase."]=Field("This key is for the decision of the LLM for showing, warning, blocking and rephrasing the the answer.")
    warning_message:str=Field("This key contain the warning given by the model if there is not any warnong then put 'None' in it.")


# creating the state 
class ai_monitoring_state(TypedDict):
    query:str
    context:str
    PDFFile_path:str

    texual_loaded_data:str
    tabular_loaded_data:str
    image_loaded_data:str
    texual_splitted_data:str
    tabular_splitted_data:str
    image_splitted_data:str
    ollama_embedding_model:any
    texual_vs:any
    tabular_vs:any
    image_vs:any
    texual_retriever_context:str
    tabular_retriever_context:str
    image_retriever_context:str

    tabular_to_text_converted_data:str

    main_prompt:str

    llm_answer:Dict
    sources:str


parser = PydanticOutputParser(
    pydantic_object=LLM_Structured_Output
)

# LLM model getting here 
model=ChatGroq(model=groq_model_name,temperature=groq_temperature)


# Creating the Document Loader node
def Texual_Data_Loader(state:ai_monitoring_state):
    texual_loader=PyPDFLoader(file_path=state["PDFFile_path"])
    texual_loaded_data=texual_loader.load()
    return {"texual_loaded_data":texual_loaded_data}

def Tabular_Data_Loader(state:ai_monitoring_state):
    tables = []
    with pdfplumber.open(state["PDFFile_path"]) as pdf:
        if pdf.pages:
            for page in pdf.pages:
                for table in page.extract_tables():
                    tables.append({"columns":table[0],"rows":table[1:]})
    return {"tabular_loaded_data":tables}

def Image_Data_Loder(state:ai_monitoring_state):
    output_dir="Images"
    
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(state["PDFFile_path"])

    image_paths = []
    image_extracted_text=[]
    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_path = f"{output_dir}/page{page_num}_img{img_index}.{image_ext}"
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(image_path)

    for i in image_paths:
        text=pytesseract.image_to_string(i)
        if text!="":
            image_extracted_text.append(text)

    image_extracted_text = "".join(c for c in image_extracted_text if c not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n')

    shutil.rmtree(output_dir)

    return {'image_loaded_data':image_extracted_text}


# node for the tabular data to the texual format creation
def Tabular_to_text_conversion(state:ai_monitoring_state):
    prompt=PromptTemplate(template="""
        You are an expert assistant specialized in transforming extracted tabular data into clean, readable natural language.
        You will receive content that originally comes from a table, but it may appear flattened, noisy, incomplete, or poorly structured.

        Your task is to:

            1. Identify the table structure by recognizing the **column headers** (under the "columns" key) and the corresponding **row values** (under the "rows" key), even if formatting is imperfect.
            2. Preserve all semantic relationships between columns and rows accurately.
            3. Convert each row into a complete, clear, and grammatically correct sentence using the column context.
            4. Ensure that no information from the original table is lost or ignored.
            5. Do NOT hallucinate, guess missing values, or introduce any new information that is not present in the data.
            6. Produce the final output as a coherent textual explanation suitable for documentation, retrieval, and downstream RAG applications.

        Tabular Data Input:
        {tabular_data}

        Return only the converted textual representation.
    """,input_variables=['tabular_data'])

    llm = ChatGroq(model=groq_model_name,temperature=groq_temperature)
    parser = StrOutputParser()
    
    chain = prompt | llm | parser
    data=chain.invoke({
        "tabular_data": state["tabular_loaded_data"]
    })
    return {'tabular_to_text_converted_data':data}


# Creating the text splitter node
def Textual_Data_Splitter(state: ai_monitoring_state):
    textual_data = state["texual_loaded_data"]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(textual_data)
    return {"texual_splitted_data": texts}

def Tabular_Data_Splitter(state:ai_monitoring_state):
    tabular_to_text_data = state["tabular_to_text_converted_data"]
    docs = [
        Document(page_content=json.dumps(table))
        for table in tabular_to_text_data
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)
    return {"tabular_splitted_data": texts}

def Image_Data_Splitter(state:ai_monitoring_state):
    image_data=state['image_loaded_data']
    docs = [
        Document(page_content=json.dumps(i))
        for i in image_data
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)
    return {"image_splitted_data": texts}


# Node for Embedding of the texual document
def Data_Embedding(state: ai_monitoring_state):
    emb=OllamaEmbeddings(model=ollama_embedding_model_name)
    return {"ollama_embedding_model":emb}


# Node for texual data in the vectorstoring 
def Texual_VectorStore(state: ai_monitoring_state):
    tvs=FAISS.from_documents(state['texual_splitted_data'],state['ollama_embedding_model'])
    return {'texual_vs':tvs}

def Tabular_VectorStore(state: ai_monitoring_state):
    tvs=FAISS.from_documents(state['tabular_splitted_data'],state['ollama_embedding_model'])
    return {'tabular_vs':tvs}

def Image_VectorStore(state:ai_monitoring_state):
    ivs=tvs=FAISS.from_documents(state['image_splitted_data'],state['ollama_embedding_model'])
    return {'image_vs':tvs}


# node for the texual data retriever
def Texual_VectorStoreRetriever(state:ai_monitoring_state):
    vs=state["texual_vs"]
    retriever=vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.1})
    out=retriever.invoke(state['query'])
    return {'texual_retriever_context':out}

def Tabular_VectorStoreRetriever(state:ai_monitoring_state):
    vs=state["tabular_vs"]
    retriever=vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.1})
    out=retriever.invoke(state['query'])
    return {'tabular_retriever_context':out}

def Image_VectorStoreRetriever(state:ai_monitoring_state):
    vs=state["image_vs"]
    retriever=vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.1})
    out=retriever.invoke(state['query'])
    return {'image_retriever_context':out}


# Node the llm answer generation
def LLM_model_call(state:ai_monitoring_state):
    
    texual_context_data="".join([i.page_content for i in state['texual_retriever_context']])
    tabular_context_data="".join([i.page_content for i in state['tabular_retriever_context']])
    image_context_data="".join([i.page_content for i in state['image_retriever_context']])

    state['main_prompt']=PromptTemplate(
        template="""
            You are the excellent output generator.
            your main work is provide the clean, polished, professional and in day to day enlgish of the user query based on the provided context.
            below is the user query and context:
            user query : {query}
            Texual context data : {texual}
            Tabular context data : {tabular}
            Image context data : {imagual}

            {format_instructions}
        """,
        input_variables=['query','texual','tabular','imagual'],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain=state['main_prompt']|model|parser

    output=chain.invoke({
        'query':state['query'],
        'texual':texual_context_data,
        'tabular':tabular_context_data,
        'imagual':image_context_data
    })

    return {"llm_answer":output}    


graph=StateGraph(ai_monitoring_state)

graph.add_node("Texual_Data_Loader",Texual_Data_Loader)
graph.add_node("Textual_Data_Splitter",Textual_Data_Splitter)
graph.add_node("Data_Embedding",Data_Embedding)
graph.add_node("Texual_VectorStore",Texual_VectorStore)
graph.add_node("Texual_VectorStoreRetriever",Texual_VectorStoreRetriever)
graph.add_node('LLM_model_call',LLM_model_call)

graph.add_node('Tabular_Data_Loader',Tabular_Data_Loader)
graph.add_node('Tabular_to_text_conversion',Tabular_to_text_conversion)
graph.add_node('Tabular_Data_Splitter',Tabular_Data_Splitter)
graph.add_node('Tabular_VectorStore',Tabular_VectorStore)
graph.add_node('Tabular_VectorStoreRetriever',Tabular_VectorStoreRetriever)

graph.add_node('Image_Data_Loder',Image_Data_Loder)
graph.add_node('Image_Data_Splitter',Image_Data_Splitter)
graph.add_node('Image_VectorStore',Image_VectorStore)
graph.add_node('Image_VectorStoreRetriever',Image_VectorStoreRetriever)


graph.add_edge(START,"Texual_Data_Loader")
graph.add_edge("Texual_Data_Loader","Textual_Data_Splitter")
graph.add_edge("Textual_Data_Splitter","Data_Embedding")
graph.add_edge("Data_Embedding","Texual_VectorStore")
graph.add_edge("Texual_VectorStore","Texual_VectorStoreRetriever")
graph.add_edge("Texual_VectorStoreRetriever",'LLM_model_call')
graph.add_edge('LLM_model_call',END)

graph.add_edge(START,"Tabular_Data_Loader")
graph.add_edge('Tabular_Data_Loader','Tabular_to_text_conversion')
graph.add_edge("Tabular_to_text_conversion",'Tabular_Data_Splitter')
graph.add_edge("Tabular_Data_Splitter","Data_Embedding")
graph.add_edge('Data_Embedding','Tabular_VectorStore')
graph.add_edge('Tabular_VectorStore','Tabular_VectorStoreRetriever')
graph.add_edge('Tabular_VectorStoreRetriever','LLM_model_call')

graph.add_edge(START,'Image_Data_Loder')
graph.add_edge('Image_Data_Loder','Image_Data_Splitter')
graph.add_edge('Image_Data_Splitter','Data_Embedding')
graph.add_edge('Data_Embedding','Image_VectorStore')
graph.add_edge('Image_VectorStore','Image_VectorStoreRetriever')
graph.add_edge('Image_VectorStoreRetriever','LLM_model_call')


workflow=graph.compile()


# output=workflow.invoke(
#     {
#         "PDFFile_path":"../Input-Document/testing.pdf",
#         "query":"What is summery of the document?"
#     }
# )

# print(output['llm_answer'].answer)

