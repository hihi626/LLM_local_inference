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
            max_new_tokens=500,
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

# Chat function
def chat_with_model(local_llm, results_path):
    """Chat with the model."""
    results = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break
        response = invoke(user_input, local_llm)
        cleaned_response = process_response(response)
        results.append({
            "input": user_input,
            "response": cleaned_response
        })
        write_results_to_file(results, results_path)

    return results

# Write results to file
def write_results_to_file(results, output_file):
    """Write the results to a text file."""
    with open(output_file, "w") as f:
        for result in results:
            f.write(f"Input: {result['input']}\n")
            f.write(f"Response: {result['response']}\n\n")
    print(f"Results saved to {output_file}")



# Test model
def test_model(local_llm):
    """Test the model with predefined test cases."""
    test_cases = [
        "What is 1+1?",
        "Who are you?"
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"Testing case {i+1}: {case}")
        response = invoke(case, local_llm)
        cleaned_response = process_response(response)
        results.append({
            "input": case,
            "response": cleaned_response
        })
    
    return results