from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from datasets import load_dataset
import torch
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
import datetime


def load_model(adapter_path=None):
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
    tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded adapter weights from {adapter_path}")
    
    return model, tokenizer

def load_model_for_training():
    model, tokenizer = load_model()
    model = prepare_model_for_kbit_training(model)

    # Add these config modifications
    model.config.use_cache = False  # Disable cache for gradient checkpointing
    model.config.pretraining_tp = 1  # For Qwen models specifically
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, peft_config), tokenizer

def prepare_training_data(filtered_data_path="filtered_dataset.jsonl"):
    dataset = load_dataset("json", data_files=filtered_data_path)["train"]
    
    def format_instruction(example):
        return {"text": f"### Input: {example['input']}\n### Response: {example['output']}"}
    
    return dataset.map(format_instruction)

def finetune_with_rag():
    model, tokenizer = load_model_for_training()
    dataset = prepare_training_data()
    
    # Modified tokenization function
    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Convert tensors to numpy arrays for dataset compatibility
        return {
            "input_ids": tokenized["input_ids"].cpu().numpy(),
            "attention_mask": tokenized["attention_mask"].cpu().numpy(),
            "labels": tokenized["input_ids"].clone().cpu().numpy()
        }
    
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "input", "output"]  # Remove all non-tensor columns
    )

    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=2,
        fp16=True,
        save_strategy="no",
        logging_steps=10,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=True,
        label_names=["labels"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # For causal language modeling
    )

    Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    ).train()

    model.save_pretrained("./finetuned_adapter")
    print("Fine-tuning complete. Adapter saved.")

def initialize_rag(adapter_path=None):
    model, tokenizer = load_model(adapter_path)
    
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
        model_kwargs={"torch_dtype": torch.float16}
    )
    return HuggingFacePipeline(pipeline=pipe)   

# Verify quantization
print("\nModel loaded successfully!")
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

def setup_rag_chain(local_llm):
    loader = TextLoader("menu.txt", encoding='utf-8')
    documents = loader.load()
    texts = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n---"
    ).split_documents(documents)

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

    return rag_chain

def invoke_rag (input_text,rag_chain):
    response = rag_chain.invoke(
        {
            "input": input_text
        }
    )
    return response

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
# testing 
# Execution Flow Control
RUN_FINETUNING = False  # Set this to True first to train the model
USE_FINETUNED = True  # Set to True to use fine-tuned adapter

if RUN_FINETUNING:
    print("\nStarting fine-tuning process...")
    finetune_with_rag()
    print("Fine-tuning completed. Adapter weights saved.\n")

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

# Initialize components
llm_pipeline = initialize_rag("./finetuned_adapter" if USE_FINETUNED else None)
rag_chain = setup_rag_chain(llm_pipeline)

# Create test function with proper arguments
def test_function(question):
    return invoke_rag(question, rag_chain)

# Run tests and save results
write_results("RAG_result.txt", questions, test_function)

print("Testing complete! Results formatted and saved.")

