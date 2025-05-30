
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




def load_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python')
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the encodings: {', '.join(encodings)}")


def get_completion_with_role(role, instruction, content):
    max_retries = 100000  # 设置最大重试次数
    for i in range(max_retries):
        try:
            # 构建结构化消息，包括角色、指令和内容
            messages = [
                {"role": "system", "content": f"You are {role}."},
                {"role": "user", "content": f"{instruction}\n{content}"}
            ]
            # 调用模型完成请求，设置温度参数
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",  # 指定使用的模型
                messages=messages,
                stream=True  # 使用流式传输
            )

            # 初始化响应内容变量
            completion_content = ""
            # 逐块获取流式响应
            for chunk in response:
                # Access content directly from chunk.choices[0].delta
                chunk_message = chunk.choices[0].delta.content
                completion_content += chunk_message
                print(chunk_message, end='', flush=True)  # 实时打印生成的内容
            return completion_content  # 修正返回变量

        except Exception as e:
            print(f"An error occurred on attempt {i + 1}: {e}")
            time.sleep(1)  # Optional: Add a delay before retrying
            continue  # Continue to retry up to `max_retries`

# def multimodal_analysis(role, instruction, image_root, tweet):
#     max_retries = 100000  # 设置最大重试次数
#     for i in range(max_retries):
#         try:
#             # 构建结构化消息，包括角色、指令和内容
#             messages = [
#                 {"role": "system", "content": f"You are a {role}."},
#                 {"role": "user", "content": [
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"{image_root}",  # Corrected f-string usage
#                             "detail": "high"
#                         }
#                     },
#                     {"type": "text", "text": f"{instruction}.\nThe sentence is: {tweet}.\n"}
#                 ]}
#             ]
#             # 调用模型完成请求，设置温度参数
#             response = client.chat.completions.create(
#                 model="Qwen/Qwen2-VL-72B-Instruct",  # 指定使用的模型
#                 messages=messages,
#                 stream=True  # 使用流式传输
#             )
#
#             # 初始化响应内容变量
#             completion_content = ""
#             # 逐块获取流式响应
#             for chunk in response:
#                 # Access content directly from chunk.choices[0].delta
#                 chunk_message = chunk.choices[0].delta.content
#                 completion_content += chunk_message
#                 print(chunk_message, end='', flush=True)  # 实时打印生成的内容
#             return completion_content  # 修正返回变量
#
#         except Exception as e:
#             print(f"An error occurred on attempt {i + 1}: {e}")
#             time.sleep(1)  # Optional: Add a delay before retrying
#             continue  # Continue to retry up to `max_retries`


###语言学大师
def linguist_analysis(tweet):
    instruction = "Accurately and concisely explain the linguistic elements in the sentence and how these elements affect meaning, including grammatical structure, tense and inflection, virtual speech, rhetorical devices, lexical choices and so on. Do nothing else."
    return get_completion_with_role("linguist", instruction, tweet)


###图形学大师
# def Image_language_expert_analysis(image_root,tweet):
#     instruction = "This is a image from a Twitter post, and the corresponding text will be provided later. Accurately describe the content of an image, including elements like characters, text, background, and setting, as well as how these elements convey or enhance the meaning or mood of the image, helping to form a complete understanding. Do nothing else."
#     return multimodal_analysis("Visual Content Analyst", instruction, image_root,tweet)


def image_judgement(tweet,word_response, image_response, target):
    judgement = get_completion_with_role(
        "multimodal content analyst. Please determine whether the image is relevant to the text based on expert analyses and the content of both the image and text, and assess whether the image can assist in stance detection for the text. Judge the relationship between the image and the text as carefully as possible. If you are unsure, please consider it irrelevant",
        f"Determine whether the image is relevant to the text.\n"
        f"The text is:{tweet},\nthe targrt is:{target}, \n"
        f"The linguist's opinion is:{word_response}.\n"
        f"The opinion of Visual Content Analyst is:{image_response}.\n"
        f"Choose from:\n A: relevant\nB:Irrelevant\n Constraint: Answer with only the option above that is most accurate and nothing else.",tweet)
    print(judgement)
    return judgement  ###prompt没改




# 加载 CSV 文件
file_path = r"dataset\RU-Conflict\zero-shot\UKR\test.csv"  # 将此处替换为您的 CSV 文件路径
data = load_csv_data(file_path)

# 检查 CSV 文件是否包含所需的列
required_columns = {'tweet_image', 'tweet_text', 'stance_target'}
if not required_columns.issubset(data.columns):
    raise ValueError(f"CSV 文件必须包含以下列: {', '.join(required_columns)}")


# 定义图像转换为 Base64 编码的函数
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# 创建用于存储结果的目录
output_dir = "analysis_results"
os.makedirs(output_dir, exist_ok=True)

# 新增一个列表，用于保存最终的结果行
final_results = []

# 遍历每一行，处理文本和图像
for index, row in data.head(38).iterrows():
    tweet = row['tweet_text']
    target = row['stance_target']
    image_response=row['gpt4v_cot_response']
    # 构建本地图像路径
    # local_image_path = rf"D:/ICME2025/Multi-Modal-Stance-Detection/Multi-Modal-Stance-Detection/2020-US-Presidential-Election/{row['tweet_image']}"
    #
    # # 将图像转换为 Base64 字符串
    # try:
    #     base64_image = image_to_base64(local_image_path)
    #     image_url = f"data:image/jpeg;base64,{base64_image}"
    # except FileNotFoundError:
    #     print(f"图像文件未找到，路径为: {local_image_path}")
    #     continue
    # except Exception as e:
    #     print(f"处理图像时发生错误，路径为 {local_image_path}，错误: {e}")
    #     continue

    # 调用语言学家分析文本
    word_response = linguist_analysis(tweet)

    # 调用图像分析专家进行图像内容分析
    # image_response = Image_language_expert_analysis(image_url,tweet)

    # 调用 image_judgement 函数生成 decision 结果
    decision = image_judgement(tweet, word_response, image_response, target)

    # 将结果保存到 txt 文件
    output_file = os.path.join(output_dir, f"analysis_{index + 1}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Tweet Text:\n")
        f.write(tweet + "\n\n")

        f.write("Stance Target:\n")
        f.write(target + "\n\n")

        f.write("Linguistic Analysis Result:\n")
        f.write(word_response + "\n\n")

        f.write("Decision:\n")
        f.write(decision + "\n\n")

    print(f"分析结果已保存到 {output_file}")

    # 保存最终结果到列表
    final_results.append({
        "final": decision
    })

# 将 final_results 列表转换为 DataFrame 并保存为 CSV 文件
final_results_df = pd.DataFrame(final_results)
final_results_df.to_csv(
    r"dataset\RU-Conflict\zero-shot\UKR\tsetneutural.csv",
     index=False, encoding="utf-8", header=True)  # Append without overwriting


print("所有分析已完成，并保存到 final_results.csv 文件中。")
