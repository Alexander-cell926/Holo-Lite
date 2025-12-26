# Holo-Lite: Hyperdimensional Computing for CPU



Holo-Lite is a "Forgotten & Undiscovered" approach to AI. It uses **Hyperdimensional Computing (HDC)** to perform learning tasks using high-dimensional algebra (10,000-bit vectors) instead of standard neural network backpropagation.

## Files
* **`demo_toy.py`**: A minimal, <50 line script. Run this to see the magic instantly. No complex setup.
* **`holo_brain.py`**: The robust engine. Contains the `HoloBrain` class with methods for `save()`, `load()`, `binarize`, and persistent memory. Use this for building real apps.

## Quick Start
1.  Run the toy:
    ```bash
    python demo_toy.py
    ```
2.  Import the engine for your own project:
    ```python
    from holo_brain.py import HoloBrain
    brain = HoloBrain()
    brain.learn("cat", "meow furry kitty")
    print(brain.predict("meow"))
    ```

## Why HDC?
* **Fast:** Runs on CPU (Intel i5/M1 friendly).
* **One-Shot:** Learns from single examples.

* **Transparent:** No black-box weights.
