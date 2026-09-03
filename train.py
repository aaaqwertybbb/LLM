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

# ==========================================
# 7. SPLIT INTO TRAINING & VALIDATION SETS
# ==========================================
# Calculate the index where the 90% mark sits
n = int(0.9 * len(data_tensor))

# The first 90% of the numerical array is for training
train_data = data_tensor[:n]

# The remaining 10% is for validating the AI's performance
val_data = data_tensor[n:]

print("-" * 30)
print(f"Training data size:   {len(train_data)} tokens")
print(f"Validation data size: {len(val_data)} tokens")

# ==========================================
# 8. CREATE INPUT AND TARGET CHUNKS (X & Y)
# ==========================================
# Look at an 8-character window at a time
block_size = 8

# Grab the first 8 characters as our input (X)
x = train_data[:block_size]

# Grab the next 8 characters, shifted forward by exactly 1 slot (Y)
y = train_data[1:block_size+1]

print("-" * 30)
print(f"Input Tensor (x):  {x}")
print(f"Target Tensor (y): {y}")
print("-" * 30)

# Visual explanation of what the AI sees character-by-character
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context.tolist()}, the target next character is: {target}")





