from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import os


def load_model():
    # Set your model path here
    model_path = "./../Llama-3.2-3B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",              
        torch_dtype=torch.bfloat16,          
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

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
local_llm = HuggingFacePipeline(pipeline=pipe)

print("\nModel loaded successfully!")
print(f"Device: {model.device}")

# Without RAG

PROMPT = PromptTemplate(
    input_variables=["input"],
    template="""
    Prompt:
    You are a precise and helpful bot, 
    Answer the user's question as accurately as possible.
    Your answers can be short, but they must be correct.
    Do not make up any information nor input anything that is not in the context.

    User_input :
    {input}
    
    Response:"""
)

chain = (
    PROMPT 
    | local_llm
)

def invoke(input_text):
    response = chain.invoke(
        {
            "input": input_text
        }
    )
    return response

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
# testing 

def process_response(response):
    """Extract the final bot response from the raw output"""
    cleaned_response = response.split("Response:")[-1].strip()
    return cleaned_response


def test_model():
    test_cases = [
        "What is the capital of France?",
        "How do I make a chocolate cake?"
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"Testing case {i+1}: {case}")
        response = invoke(case)
        # response = invoke_rag(case)  # Uncomment to use RAG
        results.append({
            "input": case,
            "response": response
        })
    
    return results

if __name__ == "__main__":
    results = test_model()
    
    # Save results to a text file
    output_file = "test_results.txt"
    with open(output_file, "w") as f:
        for result in results:
            f.write(f"Input: {result['input']}\n")
            f.write(f"Response: {result['response']}\n\n")
    print(f"Results saved to {output_file}")

print("Testing complete! Results formatted and saved.")

