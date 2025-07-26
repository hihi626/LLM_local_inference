from LLM_inference import (
    load_model,
    pipeline_llm,
    test_model
)

if __name__ == "__main__":

    # Set your model path here
    model_path = "./../Llama-3.2-3B-Instruct"
    
    model, tokenizer = load_model(model_path)
    local_llm = pipeline_llm(model, tokenizer)
    results = test_model(local_llm)
    
    # Save results to a text file
    output_file = "test_results.txt"
    with open(output_file, "w") as f:
        for result in results:
            f.write(f"Input: {result['input']}\n")
            f.write(f"Response: {result['response']}\n\n")
    print(f"Results saved to {output_file}")

print("Testing complete! Results formatted and saved.")
