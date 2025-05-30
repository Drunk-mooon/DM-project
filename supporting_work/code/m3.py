from openai import OpenAI
import pandas as pd
import time
import base64
import os
client = OpenAI(
    api_key="sk-wusosiuemxnmbzmzkzxekubgrbthilvffmdlfrtfgebhfgxd",  # Replace with your actual API KEY
    base_url="https://api.siliconflow.cn/v1"  # Set to SiliconFlow API base URL
)


# 设置输出目录
output_directory = "output_results/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 读取每一行的 Linguistic Analysis 和 Image Analysis 的内容
def read_individual_analysis(row_num):
    filename = f"analysis_results/analysis_{row_num}.txt"
    linguistic_opinion = ""

    section = None

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("Decision:"):
                continue
            if line.startswith("Linguistic Analysis Result:"):
                section = "linguistic"
                linguistic_opinion += line
            elif section == "linguistic":
                linguistic_opinion += line.strip()

    return linguistic_opinion

# 读取总文件中的 Expert Analysis, Self Analysis, 和 User Analysis
def read_combined_analysis(filename, row_num):
    expert_analysis_opinion = ""
    self_analysis_opinion = ""
    user_analysis_opinion = ""
    section = None
    found_row = False

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == f"Analysis for Row {row_num}:":
                found_row = True
                continue

            if found_row and line.startswith("Analysis for Row"):
                break

            if found_row:
                if line.startswith("Expert Analysis Opinion:"):
                    section = "expert"
                    expert_analysis_opinion = line.replace("Expert Analysis Opinion:", "").strip()
                elif line.startswith("Self Analysis Opinion:"):
                    section = "self"
                    self_analysis_opinion = line.replace("Self Analysis Opinion:", "").strip()
                elif line.startswith("User Analysis Opinion:"):
                    section = "user"
                    user_analysis_opinion = line.replace("User Analysis Opinion:", "").strip()
                elif section == "expert":
                    expert_analysis_opinion += " " + line.strip()
                elif section == "self":
                    self_analysis_opinion += " " + line.strip()
                elif section == "user":
                    user_analysis_opinion += " " + line.strip()

    return expert_analysis_opinion, self_analysis_opinion, user_analysis_opinion

# 生成用户消息的prompt
def generate_summary(row_num, combined_filename,image_analysis):
    linguistic_opinion = read_individual_analysis(row_num)
    expert_analysis_opinion, self_analysis_opinion, user_analysis_opinion = read_combined_analysis(combined_filename, row_num)

    summary_prompt = (
        f"You are provided with five experts' opinions:\n"
        f"Linguistic's Opinion: {linguistic_opinion}\n"
        f"Image Analysis: {image_analysis}\n"
        f"Expert Analysis Opinion: {expert_analysis_opinion}\n"
        f"Self Analysis Opinion: {self_analysis_opinion}\n"
        f"User Analysis Opinion: {user_analysis_opinion}\n\n"
        f"Based on these opinions, please complete the following:\n\n"
        f"1. **Extract the Common Core Insight**:\n"
        f"Summarize the common themes or agreements across all opinions provided above. Identify any shared interpretations, viewpoints, or conclusions reached by multiple experts.\n\n"
        f"2. **Identify Contradictory Perspectives**:\n"
        f"List any contrasting or contradictory perspectives among the opinions. Highlight where experts disagree, provide alternative interpretations, or show differences in focus or emphasis.\n\n"
        f"3. **Assess Topic Relevance, Neutrality, and Objectivity:**:\n"
        f"Based on the above analysis, determine whether the tweet is relevant to the topic and evaluate its neutrality. Analyze whether the tweet deviates from the core topic of discussion, presents unbiased perspectives from multiple sides, or if it is simply a factual, unbiased news report without clear bias.\n\n"
        f"Please ensure that each section is clear, objective, and reflects a thorough understanding of the opinions provided."
    )
    return summary_prompt

# 调用模型生成内容
def get_knowledge_summary(user_message, model_name="meta-llama/Meta-Llama-3.1-70B-Instruct"):
    max_retries = 10000
    system_message = "You are a expert in Opinion Evaluation and Synthesis"

    for i in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                stream=True
            )

            generation_response = ""
            conflict_response = ""
            summarize_response = ""
            current_task = "generation"

            for chunk in response:
                chunk_message = chunk.choices[0].delta.content

                if "**Extract the Common Core Insight**" in chunk_message:
                    current_task = "generation"
                elif "**Identify Contradictory Perspectives**" in chunk_message:
                    current_task = "conflict"
                elif "**Develop Your Own Summary Opinion**" in chunk_message:
                    current_task = "summarize"

                if current_task == "generation":
                    generation_response += chunk_message
                elif current_task == "conflict":
                    conflict_response += chunk_message
                elif current_task == "summarize":
                    summarize_response += chunk_message

            return {
                "generation": generation_response.strip(),
                "conflict": conflict_response.strip(),
                "summarize": summarize_response.strip()
            }

        except Exception as e:
            print(f"An error occurred on attempt {i + 1}: {e}")
            time.sleep(1)

    return {"generation": "Error", "conflict": "Error", "summarize": "Error"}


# Load CSV data
def load_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python')
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the encodings: {', '.join(encodings)}")


file_path = r"dataset\RU-Conflict\zero-shot\UKR\test.csv"  # 将此处替换为您的 CSV 文件路径
data = load_csv_data(file_path)


# 主函数
# def main():
#     combined_filename = r"D:\ICME2025\code\analysis_results_expert\combined_expert_analysis.txt"
#
#     output_directory = r"D:\ICME2025\code\analysis_results_expert\outputs"  # Set output directory
#
#     for row_num in range(1, 100):
#         # Get the row using iloc[] to fetch the row by index (row_num - 1 since Python is 0-indexed)
#         image_analysis = data.iloc[row_num - 1][
#             'gpt4v_cot_response']  # Ensure 'gpt4v_cot_response' exists in your DataFrame
#
#         user_message = generate_summary(row_num, combined_filename, image_analysis)
#         result = get_knowledge_summary(user_message)
#
#         # Output filename where the result will be saved
#         output_filename = os.path.join(output_directory, f"output_row_{row_num}.txt")
#         with open(output_filename, 'w', encoding='utf-8') as output_file:
#             output_file.write(f"Row {row_num} Analysis Summary:\n\n")
#             output_file.write(f"\n{result['generation']}\n\n")
#             output_file.write(f"\n{result['conflict']}\n\n")
#             output_file.write(f"\n{result['summarize']}\n")
#
#         print(f"Row {row_num} analysis saved to {output_filename}")
# 执行主程序
# 主函数
def main():
    combined_filename = r"analysis_results_expert\combined_expert_analysis.txt"

    # 设置输出目录


    for row_num in range(1, 39):
        # 使用 iloc[] 获取特定行的数据（由于 Python 是从 0 开始的，所以需要 row_num - 1）
        image_analysis = data.iloc[row_num - 1][
            'gpt4v_cot_response']  # 确保 'gpt4v_cot_response' 列在你的 DataFrame 中

        user_message = generate_summary(row_num, combined_filename, image_analysis)
        result = get_knowledge_summary(user_message)

        # 输出文件的路径
        output_filename = os.path.join(output_directory, f"output_row_{row_num}.txt")
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            output_file.write(f"Row {row_num} Analysis Summary:\n\n")
            output_file.write(f"\n{result['generation']}\n\n")
            output_file.write(f"\n{result['conflict']}\n\n")
            output_file.write(f"\n{result['summarize']}\n")

        print(f"Row {row_num} 的分析结果已保存到 {output_filename}")


# 执行主程序
main()



