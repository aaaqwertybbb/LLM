# Google AI:
# This single Python file will serve as the core workspace where you will load your text data, build the neural network layers, and run the training loop.

# ==========================================
# 1. READ THE DATA
# ==========================================
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# ==========================================
# 2. CREATE THE VOCABULARY
# ==========================================
# 'set(text)' extracts every unique character. 'sorted()' puts them in order.
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Unique characters (Vocabulary Size): {vocab_size}")

# ==========================================
# 3. BUILD THE LOOKUP DICTIONARIES
# ==========================================
# Maps a string character to an integer ID: e.g., {'a': 0, 'b': 1, 'c': 2}
char_to_int = { ch:i for i,ch in enumerate(chars) }

# Maps an integer ID back to a string character: e.g., {0: 'a', 1: 'b', 2: 'c'}
int_to_char = { i:ch for i,ch in enumerate(chars) }

# ==========================================
# 4. DEFINE ENCODE & DECODE FUNCTIONS
# ==========================================
# Encode: Takes a string of text, outputs a list of numbers
def encode(string_input):
    return [char_to_int[c] for c in string_input]

# Decode: Takes a list of numbers, outputs the original string of text
def decode(list_input):
    return "".join([int_to_char[i] for i in list_input])

# ==========================================
# 5. TEST IT OUT
# ==========================================
sample_phrase = "hello"
encoded_phrase = encode(sample_phrase)

print("-" * 30)
print(f"Original Text: {sample_phrase}")
print(f"Encoded Numbers: {encoded_phrase}")
print(f"Decoded Back:   {decode(encoded_phrase)}")

# ==========================================
# 6. CONVERT THE ENTIRE DATASET INTO TENSORS
# ==========================================
import torch

# Encode the entire text file into a massive list of integers
all_encoded_data = encode(text)

# Convert that list into a PyTorch Tensor (a heavy-duty data array)
# 'torch.long' specifies that these numbers are large integers (64-bit)
data_tensor = torch.tensor(all_encoded_data, dtype=torch.long)

print("-" * 30)
print(f"Dataset shape: {data_tensor.shape}")
print(f"First 10 tokens as a Tensor: {data_tensor[:10]}")
