import transformers
import torch



# Load the model
model_id = "meta-llama/Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Function to generate chatbot responses
def llama_chatbot(user_input):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )

    # Return the generated response
    return outputs[0]["generated_text"].strip()

# Interactive loop for the chatbot
print("Welcome to the LLaMA chatbot! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break
    response = llama_chatbot(user_input)
    print(f"Chatbot: {response}")