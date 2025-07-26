from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline
)
from langchain_community.llms import HuggingFacePipeline
from prompt import PROMPT
import torch

#Load Custom Model
def load_model(model_path):
    """Load the model and tokenizer"""
    try:
        model_path = model_path
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",              
            torch_dtype=torch.bfloat16,          
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

# Create pipeline for llm 
def pipeline_llm(model, tokenizer):
    """Create a text generation pipeline with the specified model and tokenizer."""
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.7,
            max_new_tokens=1000,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            device_map="auto", 
        )
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        raise
    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm

# Invoke model with prompt
def invoke(input_text, local_llm):
    """Invoke the model using the prompt."""

    chain = (
        PROMPT 
        | local_llm
    )

    response = chain.invoke(
        {
            "input": input_text
        }
    )
    return response


# Clean response
def process_response(response):
    """Extract the final bot response from the raw output"""
    cleaned_response = response.split("Response:")[-1].strip()
    return cleaned_response

# Test model
def test_model(local_llm):
    test_cases = [
        "Write a algo trading strategy for AAPL",
        "How do I make a chocolate cake?"
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"Testing case {i+1}: {case}")
        response = invoke(case, local_llm)
        cleaned_response = process_response(response)
        # response = invoke_rag(case)  # Uncomment to use RAG
        results.append({
            "input": case,
            "response": cleaned_response
        })
    
    return results


#----------------------------------------------------------------------------------------------------------------
# With RAG

# RAG_PROMPT = PromptTemplate(
#     input_variables=["input"],
#     template="""
#     Prompt:
#     You are a precise and helpful bot, 
#     Answer the user's question as accurately as possible.

#     DO NOT make up any information,
#     only use your knowledge and the context provided to you.
#     and if you don't know the answer, just say "I don't know".
#     the user's input is always correct, if you do not know a word,
#     do not try to relate the word to similar words,
#     it is a word you don't know.
#     simply answer "I don't know" as the answer.

#     Instructions:
#     - Be concise and straight to the point 
#     - answer the questions in the shortest way possible
#     - DO NOT include the prompt in the response 
#     - respond like a human
#     - DO NOT show your reasoning process or use tags like </think>
#     - Give only the final answer directly

#     User_input :
#     {input}

#     Context:
#     {context}
    
#     Response:"""
# )

# #load documents and split text 
# loader = TextLoader("menu.txt", encoding='utf-8')
# documents = loader.load()

# text_splitter = CharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50,
#     separator="\n---"
# )
# texts = text_splitter.split_documents(documents)

# #embedding
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
#     encode_kwargs={"normalize_embeddings": True} 
# )
# docs = [doc.page_content for doc in texts]
# metadatas = [{"source": f"chunk-{i}"} for i in range(len(texts))]

# #vectorstore
# vectorstore = Chroma.from_texts(
#     texts=docs,
#     embedding=embeddings,
#     metadatas=metadatas,
#     persist_directory="./chroma_db"
# )


# #retrival
# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# #rag_chain
# rag_chain = ( 
#     {"context":  lambda x: retriever.get_relevant_documents(x["input"]),
#       "input": RunnablePassthrough()}
#     | RAG_PROMPT
#     | local_llm
# )

# def invoke_rag (input_text):
#     response = rag_chain.invoke(
#         {
#             "input": input_text
#         }
#     )
#     return response

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
