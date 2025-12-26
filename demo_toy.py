# demo_toy_v2.py
# HOLO-LITE V2: Now with Cosine Similarity & Better Data
import numpy as np

print("\n--- 🧠 HOLO-LITE v2 (Smarter) ---")
print("Initializing 10,000-bit Hypervectors...")

# 1. SETUP
D = 10000
chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,:;()[]{}+-*/='\""
item_memory = {}

# Initialize random vectors for characters
for char in chars:
    item_memory[char] = np.random.choice([-1, 1], size=D)

# 2. ENCODER (Text -> Vector)
def text_to_vector(text):
    text = text.lower()
    text_vec = np.zeros(D, dtype=int)
    for i in range(len(text) - 2):
        c1, c2, c3 = text[i], text[i+1], text[i+2]
        if c1 in item_memory and c2 in item_memory and c3 in item_memory:
            # Bind & Permute (The "Holographic" Math)
            v1 = np.roll(item_memory[c1], 2)
            v2 = np.roll(item_memory[c2], 1)
            v3 = item_memory[c3]
            text_vec += v1 * v2 * v3
    return text_vec

# 3. COSINE SIMILARITY (The Fix!)
def get_similarity(vec_a, vec_b):
    # This prevents "Long text" from overpowering "Short text"
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

# 4. TRAINING (Better Data)
print("Training on multiple examples...")

# We combine multiple sentences to make the memory "Dense"
english_data = [
    "the quick brown fox jumps over the lazy dog",
    "hello my name is alex and i like ai",
    "this is a simple sentence for testing",
    "artificial intelligence is great"
]
python_data = [
    "def function(x): return x + 1",
    "import numpy as np",
    "print('hello world')",
    "for i in range(10): x = x * 2"
]

# Learn English
english_mem = np.zeros(D, dtype=int)
for sentence in english_data:
    english_mem += text_to_vector(sentence)

# Learn Python
python_mem = np.zeros(D, dtype=int)
for code in python_data:
    python_mem += text_to_vector(code)

print(" -> Knowledge Absorbed.")

# 5. TEST LOOP
print("\n--- READY! (Type 'quit' to exit) ---")

while True:
    user_input = input("\nYour Input > ")
    if user_input.lower() == "quit": break
    
    query_vec = text_to_vector(user_input)
    
    # Compare using the NEW math
    score_eng = get_similarity(query_vec, english_mem)
    score_py = get_similarity(query_vec, python_mem)
    
    print(f"   Confidence (English): {score_eng:.4f}")
    print(f"   Confidence (Python):  {score_py:.4f}")
    
    # Simple Logic to decide
    if score_eng > score_py:
        print(">> RESULT: ENGLISH TEXT 📝")
    else:
        print(">> RESULT: PYTHON CODE 🐍")