import pandas as pd
import time
import logging
import base64
import os
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key="sk-wjebrtutyfbvdjeabbfztkseajaufzacnseuzuatkjugjrjh",  # Replace with your actual API KEY
    base_url="https://api.siliconflow.cn/v1"  # Set to the SiliconFlow API base URL
)

# Load CSV data with encoding fallback
def load_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python')
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the encodings: {', '.join(encodings)}")

# Read back knowledge from a specified file path
def read_back_knowledge(index):
    file_path = f"debate_results/result_{index + 1}.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.error(f"Back knowledge file not found: {file_path}")
        return "Error: Back knowledge file not found."
    except Exception as e:
        logging.error(f"Error reading back knowledge file: {str(e)}")
        return "Error: Could not read back knowledge file."

# Convert an image file to a base64 string with the appropriate MIME type prefix
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"

# Multimodal analysis with retries
def multimodal_analysis(result, tweet, target):
    max_retries = 5  # Set a reasonable retry limit
    for i in range(max_retries):
        try:
            # Build structured messages
            messages = [
                {"role": "system", "content": "You are a Stance Evaluator. Please pay attention to discern whether the stance is neutral, off-topic, or a news narrative."},
                {"role": "user", "content": [
                    {"type": "text", "text":
                        f"Determine whether the sentence is in favor of or against {target}, or is neutral to {target}.\n"
                        f"Sentence: {tweet}\nJudge this in relation to the following arguments:\n"
                        f"{result}\nChoose from:\n A: Against\nB: Favor\nC: Neutral\n"
                        f"The focus should primarily be on the tweet itself, with any additional knowledge serving only as a reference. Please pay special attention to neutral judgment. If the tweet is stating objective facts without bias or showing obvious support or opposition emotions, regardless of whether the fact is positive or not, please consider it neutral. If there is no clear or strong emotional bias, or if the tweet is irrelevant to the topic, or Twitter content may have a strong emotional inclination but does not specifically point to a particular stance, the stance should be neutral."
                        "Constraint: Answer with only the option above that is most accurate and nothing else."

                     }
                ]}
            ]

            # Invoke model with streaming
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=messages,
                stream=True
            )

            # Collect and return streamed response
            completion_content = ""
            for chunk in response:
                chunk_message = chunk.choices[0].delta.content
                completion_content += chunk_message
                print(chunk_message, end='', flush=True)
            return completion_content

        except Exception as e:
            print(f"An error occurred on attempt {i + 1}: {e}")
            time.sleep(1)  # Delay before retrying
            continue
    return "Error: Unable to complete multimodal analysis."

# Main processing function for tweets
def process_tweet_data(data, file_path):
    dlddata = load_csv_data(
        r"dataset\RU-Conflict\zero-shot\UKR\tsetneutural.csv")
    results = []
    for (index, row), (index2, row2) in zip(data.head(38).iterrows(), dlddata.head(38).iterrows()):
        tweet = row['tweet_text']
        target = row['stance_target']
        final = row2['final']

        # Construct local image path



        # Retrieve back knowledge and perform multimodal analysis
        respond = read_back_knowledge(index)
        result = multimodal_analysis(respond, tweet, target)
        results.append(result)

    # Save the results to the DataFrame and CSV file
    results_df = pd.DataFrame(results)

    # Store the DataFrame into a CSV file
    results_df.to_csv(r"results.csv", index=False, encoding='utf-8')


# Load CSV file and process tweet data
file_path = r"dataset\RU-Conflict\zero-shot\UKR\test.csv"
data = load_csv_data(file_path)
process_tweet_data(data, file_path)

