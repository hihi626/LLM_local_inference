from LLM_inference import (
    load_model,
    pipeline_llm,
    chat_with_model
)

if __name__ == "__main__":

    # Settings
    model_path = "./../Llama-3.2-3B-Instruct" # Set your model path here
    results_path = "test_results.txt" # Set your results path here

    model, tokenizer = load_model(model_path)
    local_llm = pipeline_llm(model, tokenizer)
    results = chat_with_model(local_llm, results_path)

print("Testing complete! Results formatted and saved.")
