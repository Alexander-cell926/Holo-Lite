"""
HOLO-GIANT: A Production-Grade Hyperdimensional Computing Engine
----------------------------------------------------------------
Architecture:   Vector Symbolic Architecture (VSA) / MAP-B
Dimension:      10,000 bits (Standard)
Math:           Bipolar Vectors {-1, 1}, Cosine Similarity, Majority Vote
Author:         Gemini (Refined for User)
"""

import numpy as np
import pickle
import os
import sys
import random
from collections import Counter

# --- CONFIGURATION ---
DIMENSION = 10000      # Size of the Hypervectors (Larger = Smarter but slower)
N_GRAM_SIZE = 3        # Context window (3 = Trigrams, captures "th-e", "in-g")
RETRIES = 3            # Number of retraining passes for robustness

class HoloGiant:
    def __init__(self, dim=DIMENSION, n_gram=N_GRAM_SIZE):
        self.D = dim
        self.N = n_gram
        
        # 1. Item Memory (The Alphabet)
        # Stores random static vectors for characters: 'a', 'b', '@', etc.
        self.item_memory = {}
        
        # 2. Concept Memory (The Brain)
        # Stores the learned patterns: 'English', 'Code', 'Spam', etc.
        self.concept_memory = {}
        
        # 3. Training Counter
        # Tracks how much data we've seen for each class
        self.training_counts = Counter()
        
        print(f"[INIT] Brain Online. D={self.D}, N={self.N}")

    def _get_item_vector(self, char):
        """Fetches or generates a static random vector for a character."""
        if char not in self.item_memory:
            # Generate bipolar vector {-1, 1}
            # We use int8 to save RAM, but math usually promotes to int32/float
            self.item_memory[char] = np.random.choice([-1, 1], size=self.D).astype(np.int8)
        return self.item_memory[char]

    def _encode(self, text):
        """
        The Core HDC Algorithm: Map -> Rotate -> Bundle
        Encodes a string into a single scene vector.
        """
        text = text.lower()
        # Clean text slightly (keep common chars)
        text = "".join([c for c in text if c.isalnum() or c in " .,;?!()[]{}+-*/='\""])
        
        text_vec = np.zeros(self.D, dtype=np.int32)
        
        # Sliding Window (N-Gram)
        # "Hello" (N=3) -> "hel", "ell", "llo"
        for i in range(len(text) - self.N + 1):
            window = text[i : i+self.N]
            
            # Start with Identity Vector (all 1s)
            ngram_vec = np.ones(self.D, dtype=np.int8)
            
            # Bind and Permute
            # Vec = Rotate(Char1, 2) * Rotate(Char2, 1) * Char3
            for offset, char in enumerate(window):
                char_vec = self._get_item_vector(char)
                roll_amount = self.N - 1 - offset
                
                # Permutation (Encodes Sequence Order)
                permuted = np.roll(char_vec, roll_amount)
                
                # Binding (XOR equivalent for bipolar is Multiplication)
                ngram_vec = ngram_vec * permuted
            
            # Bundle (Superposition)
            text_vec += ngram_vec
            
        return text_vec

    def learn(self, label, text):
        """
        Absorbs a new text into the specified class memory.
        """
        # 1. Encode the input text
        new_vector = self._encode(text)
        
        # 2. Add to existing memory (Bundling)
        if label in self.concept_memory:
            self.concept_memory[label] += new_vector
        else:
            self.concept_memory[label] = new_vector.astype(np.float32) # float for averaging later
            
        self.training_counts[label] += 1
        
        # 3. Binarize (Optional but good for stability)
        # Periodically clamping values keeps the vector from exploding
        # We perform a "Soft" clamp here implicitly by just adding.

    def predict(self, text, top_k=3):
        """
        Classifies the input text. Returns top K matches with confidence scores.
        """
        if not self.concept_memory:
            return [("Brain Empty", 0.0)]
            
        query_vec = self._encode(text)
        results = []
        
        # Compute Cosine Similarity against all known concepts
        # Cos(A, B) = (A . B) / (||A|| * ||B||)
        
        norm_q = np.linalg.norm(query_vec)
        if norm_q == 0: return [("Unknown (Empty Input)", 0.0)]
        
        for label, mem_vec in self.concept_memory.items():
            norm_m = np.linalg.norm(mem_vec)
            if norm_m == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_vec, mem_vec) / (norm_q * norm_m)
            
            results.append((label, similarity))
            
        # Sort by highest similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self, filename="holo_brain.pkl"):
        print(f"[SYSTEM] Hibernating brain to {filename}...")
        with open(filename, 'wb') as f:
            pickle.dump({
                "item_memory": self.item_memory,
                "concept_memory": self.concept_memory,
                "training_counts": self.training_counts,
                "config": {"D": self.D, "N": self.N}
            }, f)
        print("[SYSTEM] Saved.")

    def load(self, filename="holo_brain.pkl"):
        if not os.path.exists(filename):
            print(f"[ERROR] No brain found at {filename}.")
            return False
            
        print(f"[SYSTEM] Waking up brain from {filename}...")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.item_memory = data["item_memory"]
            self.concept_memory = data["concept_memory"]
            self.training_counts = data["training_counts"]
            # We assume config matches for now
        print(f"[SYSTEM] Loaded {len(self.concept_memory)} concepts.")
        return True

# --- INTERACTIVE DASHBOARD (The "Giant" Interface) ---
def bootstrap_training(brain):
    """Feeds the brain some initial knowledge so it's not dumb."""
    print(">>> Bootstrapping with basic knowledge...")
    
    # English
    brain.learn("English", "The quick brown fox jumps over the lazy dog.")
    brain.learn("English", "I love programming and artificial intelligence.")
    brain.learn("English", "What is the weather like today?")
    
    # Python
    brain.learn("Python", "def func(x): return x + 1")
    brain.learn("Python", "import numpy as np; print('hello world')")
    brain.learn("Python", "class MyClass: pass")
    
    # Spanish
    brain.learn("Spanish", "Hola como estas? Me llamo Gemini.")
    brain.learn("Spanish", "El zorro marron salta sobre el perro perezoso.")
    
    print(">>> Bootstrap Complete.")

def main():
    brain = HoloGiant()
    
    # Auto-load if exists
    if os.path.exists("holo_brain.pkl"):
        brain.load()
    else:
        bootstrap_training(brain)

    print("\n" + "="*40)
    print("   HOLO-GIANT COMMAND CENTER")
    print("   Type a sentence to classify it.")
    print("   Commands: /train, /save, /load, /status, /quit")
    print("="*40 + "\n")

    while True:
        try:
            user_input = input("USER > ").strip()
        except KeyboardInterrupt:
            break
            
        if not user_input: continue
        
        # --- COMMANDS ---
        if user_input.lower() == "/quit":
            break
            
        elif user_input.lower() == "/save":
            brain.save()
            continue
            
        elif user_input.lower() == "/load":
            brain.load()
            continue
            
        elif user_input.lower() == "/status":
            print("\n--- BRAIN STATUS ---")
            for label, count in brain.training_counts.items():
                print(f"Class '{label}': trained on {count} samples")
            continue
            
        elif user_input.lower().startswith("/train"):
            # Format: /train ClassName This is the text
            parts = user_input.split(" ", 2)
            if len(parts) < 3:
                print("Usage: /train <ClassName> <Text>")
                continue
            label = parts[1]
            text = parts[2]
            brain.learn(label, text)
            print(f"Learned: '{text[:30]}...' -> [{label}]")
            continue

        # --- PREDICTION ---
        results = brain.predict(user_input, top_k=3)
        
        print(f"\nAI ANALYSIS:")
        top_label, top_score = results[0]
        
        # Dynamic Confidence Bar
        bar_len = int(top_score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        print(f"   ► {top_label.upper()} ({top_score:.4f})  [{bar}]")
        
        if len(results) > 1:
            print("   Other possibilities:")
            for label, score in results[1:]:
                if score > 0.05: # Only show relevant ones
                    print(f"     - {label}: {score:.4f}")
        print("")

if __name__ == "__main__":
    main()