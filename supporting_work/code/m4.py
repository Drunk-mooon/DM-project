from openai import OpenAI
import pandas as pd
import time
import logging
import base64
import os

client = OpenAI(
    api_key="sk-wjebrtutyfbvdjeabbfztkseajaufzacnseuzuatkjugjrjh",  # 替换为您的实际 API KEY
    base_url="https://api.siliconflow.cn/v1"  # 设置为 SiliconFlow 的 API 基础 URL
)

# 设置输出目录
output_dir = "debate_results"
os.makedirs(output_dir, exist_ok=True)

# 定义图像转换为 Base64 编码的函数
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 读取每一行的 back knowledge 从对应的 .txt 文件
def read_back_knowledge(index):
    try:
        file_path = f"output_results/output_row_{index+1}.txt"
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.error(f"Back knowledge file not found: {file_path}")
        return "Error: Back knowledge file not found."
    except Exception as e:
        logging.error(f"Error reading back knowledge file: {str(e)}")
        return "Error: Could not read back knowledge file."

# 加载 CSV 文件
def load_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python')
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the encodings: {', '.join(encodings)}")

# 定义 stance_analysis 函数
def stance_analysis(tweet, result, target, max_retries=10000):
    system_messages = {
        'favor_debate': "You are a Ukrainian. Based on the information below, analyze the stance towards Ukraine {favor, against, neutral}. First, determine if there is any explicit support or opposition. If neither is explicit, classify the stance as neutral. Provide specific textual evidence for your conclusion, and explain why the evidence supports your classification.",
        'against_debate': "You are a political scientist. Based on the information below, analyze the stance towards Ukraine {favor, against, neutral}. Carefully evaluate the language and tone for signs of support or opposition. If no clear bias is evident, classify the stance as neutral. Provide evidence from the text and explain how it supports your analysis.",
        'neutral_debate': "You are a heavy social media user. Based on the information below, analyze the stance towards Ukraine {favor, against, neutral}. Pay close attention to the tone and language used. If the stance is not clearly favoring or opposing Ukraine, classify it as neutral. Always provide text-based evidence to support your decision."
    }

    user_message = (
        f"Evaluate the stance on '{target}' based on the following content:\n"
        f"Tweet: {tweet}\n"
        f"Expert opinion: {result}\n"
        # f"Based on this information, provide arguments for the conclusion:\n"
    )

    responses = {"favor": "", "against": "", "neutral": ""}

    for stance in ["favor", "against", "neutral"]:
        for attempt in range(max_retries):
            system_message = system_messages[f"{stance}_debate"]

            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]

                response = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=messages,
                    temperature=0.1,
                    stream=True
                )

                debate_response = ""
                for chunk in response:
                    chunk_message = chunk.choices[0].delta.content
                    debate_response += chunk_message
                    print(chunk_message, end='', flush=True)

                responses[stance] = debate_response.strip()
                break  # 跳出重试循环
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}: Error generating response for {stance} debate: {str(e)}")
                time.sleep(1)

    return responses

# 定义 mmstance_analysis 函数
def mmstance_analysis(tweet, result, target, image_response, max_retries=10000):
    system_messages = {
        'favor_debate': "You are a Ukrainian. Please analyse the stance towards Ukraine{favor,against,neutral} based on the information below,pay attention to tweet please provide your evidence.",
        'against_debate': "You are political scientist. Please analyse the stance towards Ukraine{favor,against,neutral} based on the information below,pay attention to tweet please provide your evidence.",
        'neutral_debate': "You are heavy social media user. Please analyse the stance towards Ukraine{favor,against,neutral} based on the information below,pay attention to tweet please provide your evidence."
    }

    user_message = (
        f"Evaluate the stance on '{target}' based on the following content:\n"
        f"Tweet: {tweet}\n"
        f"Expert opinion: {result}\n"
        f"Based on this information, provide arguments for the conclusion:\n"
        f"if not neutral,please provide your evidence:\n"
    )

    responses = {"favor": "", "against": "", "neutral": ""}

    for stance in ["favor", "against", "neutral"]:
        for attempt in range(max_retries):
            system_message = system_messages[f"{stance}_debate"]

            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": [
                        # {
                        #     "type": "image_url",
                        #     "image_url": {
                        #         "url": image_url,
                        #         "detail": "high"  # 高分辨率图像
                        #     }
                        # },
                        {"type": "text", "text": f"{user_message}\nThe image content is:{image_response}"}
                    ]}
                ]

                response = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=messages,
                    temperature=0.1,
                    stream=True
                )

                debate_response = ""
                for chunk in response:
                    chunk_message = chunk.choices[0].delta.content
                    debate_response += chunk_message
                    print(chunk_message, end='', flush=True)

                responses[stance] = debate_response.strip()
                break  # 跳出重试循环
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}: Error generating response for {stance} debate: {str(e)}")
                time.sleep(1)

    return responses

# 定义主处理函数
def process_tweet_data(data):
    dlddata= load_csv_data(r"dataset\RU-Conflict\zero-shot\UKR\tsetneutural.csv",)
    for (index, row), (index2, row2) in zip(data.head(38).iterrows(), dlddata.head(38).iterrows()):
        tweet = row['tweet_text']
        target = row['stance_target']
        final = row2['final']
        image_response = row['gpt4v_cot_response']
        result = read_back_knowledge(index)

        # local_image_path = rf"D:/ICME2025/Multi-Modal-Stance-Detection/Multi-Modal-Stance-Detection/2020-US-Presidential-Election/{row['tweet_image']}"
        # try:
        #     base64_image = image_to_base64(local_image_path)
        #     image_url = f"data:image/jpeg;base64,{base64_image}"
        # except FileNotFoundError:
        #     print(f"Image file not found at path: {local_image_path}")
        #     continue
        # except Exception as e:
        #     print(f"Error processing image at path {local_image_path}: {e}")
        #     continue

        response = stance_analysis(tweet, result, target) if (("irrelevant" in final or "B" in final)) else mmstance_analysis(tweet, result, target, image_response)

        output_file = os.path.join(output_dir, f"result_{index + 1}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Favor Response:\n{response['favor']}\n\n")
            # f.write(f"Against Response:\n{response['against']}\n\n")
            # f.write(f"Neutral Response:\n{response['neutral']}\n\n")

        print(f"Analysis results saved to {output_file}")

# 加载 CSV 文件并运行主处理函数
file_path = r"dataset\RU-Conflict\zero-shot\UKR\test.csv"
data = load_csv_data(file_path)
process_tweet_data(data)
