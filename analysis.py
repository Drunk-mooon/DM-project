import json
from datetime import datetime
from gpt_process import chat_with_gpt
import os
import numpy as np

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
    Sort user posts chronologically and format them as text
    """
    # Sort by timestamp
    sorted_posts = sorted(posts, key=lambda x: x['time'])
    
    # Build analysis text
    analysis_text = ""
    for post in sorted_posts:
        timestamp = datetime.fromtimestamp(post['time'])
        analysis_text += f"[{timestamp}] {post['text']}\n"
    
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
    Analyze all users in the JSON file and update original data
    """
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    # Iterate and update user data
    for user in data:
        user_id = user['user_id']
        user_name = user['user_name']
        print(f"\nAnalyzing user {user_name} (ID: {user_id})...")
        
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
            print(f"Analysis complete for {user_name}")
            print(f"Personality vector: {personality_vector}")
    
    # Save updated original data
    output_path = os.path.join(BASE_DIR, "data", "reddit_user_posts_filtered_updated.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return results

if __name__ == "__main__":
    json_path = os.path.join(BASE_DIR, "data", "reddit_user_posts_filtered.json")
    
    # Run stability test on first user
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if data:
        first_user = data[0]
        print(f"Running stability test for user: {first_user['user_name']}")
        stability_stats = test_personality_stability(first_user['posts'])
        
        # Save stability test results
        if stability_stats:
            with open('personality_stability_test.json', 'w') as f:
                json.dump(stability_stats, f, indent=2)
    
    # # Run normal analysis
    # results = analyze_all_users(json_path)
    # with open('user_personality_vectors.json', 'w') as f:
    #     json.dump(results, f, indent=2)