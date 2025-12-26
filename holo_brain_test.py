# holo_brain_complex.py
# A robust, object-oriented HDC Library for Developers.
# Features: Persistency, Binarization, Cosine Similarity, Dynamic Memory.

import numpy as np
import pickle
import os

class HoloBrain:
    def __init__(self, dim=10000, n_gram=3):
        """
        Initialize the Holographic Brain.
        dim: Dimension of hypervectors (default 10,000)
        n_gram: Size of the sliding window for sequence encoding (default 3)
        """
        self.D = dim
        self.N = n_gram
        self.item_memory = {}  # Char -> Vector Mapping
        self.concept_memory = {} # Label -> Learned Vector Mapping
        self.is_trained = False
        
        # Seed for reproducibility (optional)
        np.random.seed(42)

    def _get_item_vector(self, char):
        """Lazy loading of character vectors."""
        if char not in self.item_memory:
            self.item_memory[char] = np.random.choice([-1, 1], size=self.D).astype(np.int8)
        return self.item_memory[char]

    def encode(self, text, binarize=False):
        """
        Encodes text into a hypervector.
        Args:
            text (str): Input text.
            binarize (bool): If True, clamps output to -1 or 1 (Majority Vote).
                             Useful for keeping memory clean over long training.
        """
        text = text.lower()
        text_vec = np.zeros(self.D, dtype=np.int32) # Use int32 to prevent overflow during summing
        
        # Sliding window encoding
        for i in range(len(text) - self.N + 1):
            window = text[i : i+self.N]
            
            # Create the n-gram vector via Binding (XOR/Multiplication) and Permutation (Roll)
            # Example for 3-gram: Roll(A, 2) * Roll(B, 1) * C
            ngram_vec = np.ones(self.D, dtype=np.int8)
            
            for offset, char in enumerate(window):
                roll_amount = self.N - 1 - offset
                char_vec = self._get_item_vector(char)
                permuted_vec = np.roll(char_vec, roll_amount)
                ngram_vec = ngram_vec * permuted_vec
            
            # Bundle (Superposition)
            text_vec += ngram_vec
            
        if binarize:
            # Thresholding function (Majority Vote)
            text_vec = np.where(text_vec > 0, 1, -1)
            
        return text_vec

    def learn(self, label, text):
        """
        Associates text with a label.
        Accumulates knowledge (Bundling) if label already exists.
        """
        print(f"[HOLO] Learning pattern for '{label}'...")
        vector = self.encode(text)
        
        if label in self.concept_memory:
            self.concept_memory[label] += vector
        else:
            self.concept_memory[label] = vector
            
        self.is_trained = True

    def predict(self, text, return_scores=False):
        """
        Classifies input text against learned concepts using Cosine Similarity.
        """
        if not self.is_trained:
            return "Error: Brain is empty. Train me first!"
            
        query_vec = self.encode(text)
        best_score = -np.inf
        best_label = None
        all_scores = {}
        
        for label, mem_vec in self.concept_memory.items():
            # Cosine Similarity = (A . B) / (||A|| * ||B||)
            dot_product = np.dot(query_vec, mem_vec)
            norm_q = np.linalg.norm(query_vec)
            norm_m = np.linalg.norm(mem_vec)
            
            if norm_q == 0 or norm_m == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_q * norm_m)
                
            all_scores[label] = round(float(similarity), 4)
            
            if similarity > best_score:
                best_score = similarity
                best_label = label
                
        if return_scores:
            return best_label, all_scores
        return best_label

    def save(self, filename="holo_brain.pkl"):
        """Saves the entire brain state to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"[SYSTEM] Brain saved to {filename}")

    def load(self, filename="holo_brain.pkl"):
        """Loads a brain state from a file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.__dict__ = pickle.load(f)
            print(f"[SYSTEM] Brain loaded from {filename}")
        else:
            print(f"[ERROR] File {filename} not found.")

# --- Developer Usage Example (runs only if executed directly) ---
if __name__ == "__main__":
    # Simulate a developer using the library
    brain = HoloBrain()
    
    # Train
    brain.learn("positive", "This is great amazing wonderful happy good success")
    brain.learn("negative", "This is bad terrible error fail sad horrible")
    
    # Save
    brain.save()
    
    # Predict
    result, scores = brain.predict("I feel amazing today", return_scores=True)
    print(f"Prediction: {result}")
    print(f"Scores: {scores}")