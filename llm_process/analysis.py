import json
from datetime import datetime
from gpt_process import chat_with_gpt
import os
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_personality_stability(user_posts, num_tests=5):
    """
    Analyze personality vector stability by running multiple tests on the same user
    Returns statistical metrics for each dimension
    """
    # Store all vectors
    vectors = []
    
    print(f"\nRunning {num_tests} tests for stability analysis...")
    for i in range(num_tests):
        print(f"Test {i+1}/{num_tests}...")
        vector = get_personality_vector(user_posts)
        if vector:
            vectors.append(vector)
    
    if not vectors:
        print("No valid vectors generated")
        return None
    
    # Convert to numpy array for easier calculation
    vectors = np.array(vectors)
    
    # Calculate statistics for each dimension
    stats = {}
    dimension_names = ["Politeness", "Vulgarity", "Filler Words", "Concept Density", 
                      "Social Tendency", "Influence", "Interaction Frequency", 
                      "Controversy", "Extraversion", "Agreeableness", "Conscientiousness",
                      "Neuroticism", "Openness", "Analytical", "Insight", "Causality",
                      "Certainty", "Hesitancy"]
    for i, dim_name in enumerate(dimension_names):
        dim_values = vectors[:, i]
        stats[dim_name] = {
            'mean': np.mean(dim_values),
            'std': np.std(dim_values),
            'range': np.ptp(dim_values),  # Peak to peak (max - min)
            'min': np.min(dim_values),
            'max': np.max(dim_values)
        }
    
    # Print summary
    print("\nStability Analysis Results:")
    print("-" * 50)
    for dim, metrics in stats.items():
        print(f"\n{dim}:")
        print(f"  Mean: {metrics['mean']:.2f}")
        print(f"  Std Dev: {metrics['std']:.2f}")
        print(f"  Range: {metrics['range']} (min: {metrics['min']}, max: {metrics['max']})")
    
    return stats

def analyze_user_posts(posts):
    """
    Sort user posts chronologically and format them as text with additional metadata
    """
    # Sort by timestamp
    sorted_posts = sorted(posts, key=lambda x: x['time'])
    
    # Build analysis text
    analysis_text = ""
    for post in sorted_posts:
        timestamp = datetime.fromtimestamp(post['time'])
        analysis_text += f"[{timestamp}] {post['text']}\n"
        analysis_text += f"    Votes: +{post['ups']:.0f}/-{post['downs']:.0f} | "
        analysis_text += f"Author Stats: Link Karma: {post['authorlinkkarma']}, "
        analysis_text += f"Comment Karma: {post['authorkarma']}, "
        analysis_text += f"Gold Status: {'Yes' if post['authorisgold'] else 'No'}\n"
    
    return analysis_text

def get_personality_vector(user_posts):
    """
    Analyze user personality using GPT and return an 18-dimensional feature vector
    """
    # Prepare analysis text
    analysis_text = analyze_user_posts(user_posts)
    
    # Build GPT prompt
    prompt = f"""Analyze the following user's posts and generate an 18-dimensional personality feature vector.
Rate each dimension from 0-5, where 0 indicates the weakest and 5 indicates the strongest trait.
Dimensions include: Politeness, Vulgarity, Filler Words Usage, Concept Density, Social Tendency, 
Influence, Interaction Frequency, Controversy Tendency, Extraversion, Agreeableness, Conscientiousness, 
Neuroticism, Openness, Analytical Thinking, Insight, Causality, Certainty, Hesitancy.

User posts:
{analysis_text}

Please return an 18-number vector directly, with numbers separated by commas. For example:
3,2,4,1,5,2,3,1,4,5,2,1,3,4,2,3,5,1
"""
    
    # Call GPT for analysis
    response = chat_with_gpt(prompt)
    
    try:
        # Parse GPT response vector
        vector = [int(x.strip()) for x in response.strip().split(',')]
        if len(vector) != 18:
            raise ValueError("Vector length is not 18")
        return vector
    except Exception as e:
        print(f"Error parsing GPT response: {e}")
        return None

def analyze_all_users(json_path):
    """
    Analyze all users in the JSON file and update original data with progress bar
    and periodic saving
    """
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    save_interval = 10  # 每处理10个用户保存一次
    last_save = 0
    total_users = 1000

    data = data[:total_users]  # 限制处理的用户数量
    
    # 使用tqdm创建进度条
    pbar = tqdm(data, desc="分析用户", unit="user")
    
    # Iterate and update user data
    for i, user in enumerate(pbar):
        user_id = user['user_id']
        user_name = user['user_name']
        pbar.set_description(f"正在分析用户{user_id}: {user_name}")
        
        # Get personality vector
        personality_vector = get_personality_vector(user['posts'])
        
        if personality_vector:
            # Update results dictionary
            results[user_id] = {
                'name': user_name,
                'vector': personality_vector
            }
            # Update user info in original data
            user['personality_vector'] = personality_vector
        
        # 每处理save_interval个用户就保存一次数据
        if (i + 1) % save_interval == 0:
            output_path = os.path.join(BASE_DIR, "data", "reddit_user_posts_filtered_updated.json")
            temp_output_path = output_path + '.temp'  # 使用临时文件
            
            try:
                # 先写入临时文件
                with open(temp_output_path, 'w', encoding='utf-8') as f:
                    json.dump(data[:i+1], f, indent=2, ensure_ascii=False)
                
                # 如果写入成功，替换原文件
                if os.path.exists(output_path):
                    os.replace(temp_output_path, output_path)
                else:
                    os.rename(temp_output_path, output_path)
                
                # 同时保存当前的results
                with open('user_personality_vectors.json.temp', 'w') as f:
                    json.dump(results, f, indent=2)
                
                if os.path.exists('user_personality_vectors.json'):
                    os.replace('user_personality_vectors.json.temp', 'user_personality_vectors.json')
                else:
                    os.rename('user_personality_vectors.json.temp', 'user_personality_vectors.json')
                
                last_save = i + 1
                pbar.set_description(f"已保存进度 - 已处理{last_save}个用户")
            except Exception as e:
                print(f"\n保存进度时出错: {e}")
    
    # 保存最终结果
    output_path = os.path.join(BASE_DIR, "data", "reddit_user_posts_filtered_updated.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return results

if __name__ == "__main__":
    json_path = os.path.join(BASE_DIR, "data", "reddit_user_posts_filtered.json")
    try:
        results = analyze_all_users(json_path)
        # 最终保存结果
        with open('user_personality_vectors.json', 'w') as f:
            json.dump(results, f, indent=2)
    except KeyboardInterrupt:
        print("\n程序被用户中断，已保存最近的处理结果")
    except Exception as e:
        print(f"\n程序出现异常: {e}")