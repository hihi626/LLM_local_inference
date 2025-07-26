from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
import datetime


def load_model():
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    
    # 4-bit config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.float16   # Mixed precision for better performance
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.7,
    max_new_tokens= 1000,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.1,
    device_map="auto", 
)
local_llm = HuggingFacePipeline(pipeline=pipe)

# Verify quantization
print("\nModel loaded successfully!")
print(f"Device: {model.device}")

# Without RAG

PROMPT = PromptTemplate(
    input_variables=["input"],
    template="""
    Prompt:
    You are a precise and helpful bot, 
    Answer the user's question as accurately as possible.

    DO NOT make up any information,
    only use your knowledge and the context provided to you.
    and if you don't know the answer, just say "I don't know".
    the user's input is always correct, if you do not know a word,
    do not try to relate the word to similar words,
    it is a word you don't know.
    simply answer "I don't know" as the answer.

    Instructions:
    - Be concise and straight to the point 
    - answer the questions in the shortest way possible
    - DO NOT include the prompt in the response 
    - respond like a human
    - DO NOT show your reasoning process or use tags like </think>
    - Give only the final answer directly

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

RAG_PROMPT = PromptTemplate(
    input_variables=["input"],
    template="""
    Prompt:
    You are a precise and helpful bot, 
    Answer the user's question as accurately as possible.

    DO NOT make up any information,
    only use your knowledge and the context provided to you.
    and if you don't know the answer, just say "I don't know".
    the user's input is always correct, if you do not know a word,
    do not try to relate the word to similar words,
    it is a word you don't know.
    simply answer "I don't know" as the answer.

    Instructions:
    - Be concise and straight to the point 
    - answer the questions in the shortest way possible
    - DO NOT include the prompt in the response 
    - respond like a human
    - DO NOT show your reasoning process or use tags like </think>
    - Give only the final answer directly

    User_input :
    {input}

    Context:
    {context}
    
    Response:"""
)

#load documents and split text 
loader = TextLoader("menu.txt", encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n---"
)
texts = text_splitter.split_documents(documents)

#embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True} 
)
docs = [doc.page_content for doc in texts]
metadatas = [{"source": f"chunk-{i}"} for i in range(len(texts))]

#vectorstore
vectorstore = Chroma.from_texts(
    texts=docs,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="./chroma_db"
)


#retrival
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


#rag_chain
rag_chain = ( 
    {"context":  lambda x: retriever.get_relevant_documents(x["input"]),
      "input": RunnablePassthrough()}
    | RAG_PROMPT
    | local_llm
)

def invoke_rag (input_text):
    response = rag_chain.invoke(
        {
            "input": input_text
        }
    )
    return response

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
# testing 

#questinon list
questions = [
    "Are you aware of a restaurant called the HIDDEN GEM RESTAURANT?",
    "What's the price of the Quantum Tofu Bowl the HIDDEN GEM RESTAURANT?",
    "Where is the HIDDEN GEM RESTAURANT located at ?",
    "Which days can I get Wandering Dumpling Soup in the HIDDEN GEM RESTAURANT?",
    "What is the secret item offered by the HIDDEN GEM RESTAURANT?",
    "Could I order Foggy Morning Pancakes at 2 PM at the HIDDEN GEM RESTAURANT?",
    "What's the cheapest lunch/dinner option in the HIDDEN GEM RESTAURANT?",
    "List all dishes containing truffle elements in the HIDDEN GEM RESTAURANT?",
    "What phrase is the motto the HIDDEN GEM RESTAURANT?",
    "Is there a kids menu available the HIDDEN GEM RESTAURANT?"
]

def process_response(response):
    """Extract the final bot response from the raw output"""
    bot_response = response.split("Response:")[-1].strip()
    cleaned_response = bot_response.split("</think>")[-1].strip()
    return cleaned_response

def write_results(filename, questions, test_function):
    """Write formatted results to a file"""
    with open(filename, 'w') as f:
        f.write(f"=== Test Results ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")
        
        for i, question in enumerate(questions, 1):
            responses = [process_response(test_function(question)) for _ in range(3)]
            
            f.write(f"Question {i}: {question}\n")
            for j, response in enumerate(responses, 1):
                f.write(f"Answer {j}: {response}\n")
            f.write("\n")  # Add space between questions

# Run tests and save results
write_results("No_RAG_result.txt", questions, invoke)
write_results("RAG_result.txt", questions, invoke_rag)

print("Testing complete! Results formatted and saved.")

