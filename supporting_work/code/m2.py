
from openai import OpenAI
import pandas as pd
import time
import base64
import os

# Initialize OpenAI client
client = OpenAI(
    api_key="sk-wjebrtutyfbvdjeabbfztkseajaufzacnseuzuatkjugjrjh",  # Replace with your actual API KEY
    base_url="https://api.siliconflow.cn/v1"  # Set to SiliconFlow API base URL
)

# Load CSV data
def load_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python')
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the encodings: {', '.join(encodings)}")

# Define expert, self, and user analysis functions
def relevantexpert_analysis(tweet, target, image_response):
    role = "political scientist"
    instruction = f"Please analyze the sentence in relation to the image content and determine the stance of both towards {target}, providing a concise explanation."
    return multimodal_analysis(role, instruction, image_response, tweet)

def relevantself_analysis(tweet, target,image_response):
    role = "A Ukrainian"
    instruction = f"Please analyze the sentence in relation to the image content and determine the stance of both towards {target}, providing a concise explanation."
    return multimodal_analysis(role, instruction, image_response, tweet)

def relevantuser_analysis(tweet, image_response):
    instruction = "Please analyze the sentence in relation to the image content and determine the stance of the content, providing a concise explanation."
    return multimodal_analysis("heavy social media user", instruction,image_response, tweet)

def expert_analysis(tweet, target):
    role = "political scientist"
    instruction = f"Please analyze the sentence and determine the stance of both towards {target}, providing a concise explanation."
    return get_completion_with_role(role, instruction, tweet)

def self_analysis(tweet, target):
    role = "A Ukrainian"
    instruction = f"Please analyze the sentence and determine the stance of both towards {target}, providing a concise explanation."
    return get_completion_with_role(role, instruction, tweet)

def user_analysis(tweet):
    instruction = "Please analyze the sentence and determine the stance of the content, providing a concise explanation."
    return get_completion_with_role("heavy social media user", instruction, tweet)

# Basic model call for text analysis
def get_completion_with_role(role, instruction, content):
    max_retries = 10  # Set max retries to 3
    for i in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": f"You are {role}."},
                {"role": "user", "content": f"{instruction}\n{content}"}
            ]
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=messages,
                stream=True
            )
            completion_content = ""
            for chunk in response:
                chunk_message = chunk.choices[0].delta.content
                completion_content += chunk_message
                print(chunk_message, end='', flush=True)
            return completion_content
        except Exception as e:
            print(f"An error occurred on attempt {i + 1}: {e}")
            time.sleep(1)
            continue

# Multimodal analysis for image and text
def multimodal_analysis(role, instruction, image_response, tweet):
    max_retries = 10  # Set max retries to 3
    for i in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": f"You are {role}."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"{instruction}.\nThe sentence is: {tweet}.\nThe image content is: {image_response}"}
                ]}
            ]
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=messages,
                stream=True
            )
            completion_content = ""
            for chunk in response:
                chunk_message = chunk.choices[0].delta.content
                completion_content += chunk_message
                print(chunk_message, end='', flush=True)
            return completion_content
        except Exception as e:
            print(f"An error occurred on attempt {i + 1}: {e}")
            time.sleep(1)
            continue

# Convert image to Base64 encoding
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Process data and generate combined result
file_path =r"dataset\RU-Conflict\zero-shot\UKR\test.csv"
data = load_csv_data(file_path)

required_columns = {'tweet_image', 'tweet_text', 'stance_target'}
if not required_columns.issubset(data.columns):
    raise ValueError(f"CSV file must contain the following columns: {', '.join(required_columns)}")

output_dir = "analysis_results_expert"
os.makedirs(output_dir, exist_ok=True)

# Define the combined output file path
output_file_path = os.path.join(output_dir, "combined_expert_analysis.txt")

# Open a single output file for all results
with open(output_file_path, "w", encoding="utf-8") as output_file:
    dlddata = load_csv_data(
        r"dataset\RU-Conflict\zero-shot\UKR\tsetneutural.csv")
    for (index, row), (index2, row2) in zip(data.head(38).iterrows(), dlddata.head(38).iterrows()):
        tweet = row['tweet_text']
        target = row['stance_target']
        decision = row2['final']
        image_response=row['gpt4v_cot_response']
        if ("irrelevant" in decision or "B" in decision):
            expopinion = expert_analysis(tweet, target)
            self_opinion = self_analysis(tweet, target)
            user_opinion = user_analysis(tweet)
        else:

            expopinion = relevantexpert_analysis(tweet, target, image_response)
            self_opinion = relevantself_analysis(tweet, target, image_response)
            user_opinion = relevantuser_analysis(tweet, image_response)

        # Write the results for each row into the combined output file
        output_file.write(f"Analysis for Row {index + 1}:\n")
        output_file.write("Tweet Text:\n")
        output_file.write(tweet + "\n\n")

        output_file.write("Stance Target:\n")
        output_file.write(target + "\n\n")

        output_file.write("Expert Analysis Opinion:\n")
        output_file.write(expopinion + "\n\n")

        output_file.write("Self Analysis Opinion:\n")
        output_file.write(self_opinion + "\n\n")

        output_file.write("User Analysis Opinion:\n")
        output_file.write(user_opinion + "\n\n")
        output_file.write("="*50 + "\n\n")  # Separator for each row's output

print(f"All analyses have been completed and saved in {output_file_path}")