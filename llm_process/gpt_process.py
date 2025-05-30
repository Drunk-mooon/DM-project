from openai import OpenAI

client = OpenAI()

def chat_with_gpt(prompt):
    """
    Function to send a prompt to the GPT model and receive a response.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Specify the model you want to use
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        # Extract and return the assistant's reply
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    user_input = "hello"
    reply = chat_with_gpt(user_input)
    print("GPT reply:", reply)