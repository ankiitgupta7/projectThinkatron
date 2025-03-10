# import os
# import openai

# client = openai.OpenAI(
#     api_key=os.environ["DEEPSEEK_API_KEY"],
#     base_url="https://api.deepseek.com/v1"
# )

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[{"role": "user", "content": "Hello, DeepSeek!"}]
# )

# print(response)


from ai_scientist.llm import create_client

# Use your DeepSeek model (ensure your DEEPSEEK_API_KEY is set in the environment)
client, client_model = create_client("deepseek-chat")
print("Client model:", client_model)

# Prepare a simple prompt
prompt = "Hello, please say something interesting."

# Make a sample API call using the chat completions method.
response = client.chat.completions.create(
    model=client_model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.7,
    max_tokens=50,
    n=1,
    stop=None,
)

# Print the response from the API.
print("Response:", response)
