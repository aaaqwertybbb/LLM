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
chars = sorted(list(set(text))) # 'set' as in "set theory", 'sorted' probably takes a 'list' type. And 'list' type has a 'constructor' that takes a 'set' type.
vocab_size = len(chars)
print(f"Unique characters (Vocabulary Size): {vocab_size}")

# ==========================================
# 3. BUILD THE LOOKUP DICTIONARIES
# ==========================================
# Maps a string character to an integer ID: e.g., {'a': 0, 'b': 1, 'c': 2}
char_to_int = { ch:i for i,ch in enumerate(chars) }
# ^
#
# (SPECULATION...
#  this is all my own speculation on what I'm reading here, and I have no idea what this actually means)
#
# 'enumerate(chars)' returns a python 'enumerable'.
# Each loop of the 'for ... in ...' python expression provides variables containing data, depending on the amount of variables you ask for, and whether there is known to be something returnable to that nth variable.
# A 'one variable declared' of 'for ... in ...' I presume will return 'i' (the index).
# And presumably this 'two variable declared' I presume will return 'i' (the index) and 'ch' (the character at that index within the enumerable)
# And perhaps even more variables being declared would be permissible and result in the variable having actual data, I'm not sure.
#
# The python syntax { ... } presumably allows you to create an arbitrary object with any properties you want.
# And likely python supports you providing a pattern, and then the arbitrary object will receive properties programmatically based on your pattern.
#
# So the pattern here is:
# 'ch:i for i,ch in enumerate(chars)'
#
# I think I can rewrite this as:
# 'ch:i (for i,ch in enumerate(chars))'
# (^ so that I can emphasize the implicit operator, and that there are two operands: 'chi:i', and 'for i,ch in enumerate(chars)').
# 
# Perhaps you can provide either a function to be "applied" or you can write a code block when writing 'for i,ch in enumerate(chars)'.
# So here, I presume they're applying a pattern to 'for i,ch in enumerate(chars)'.
# So this is contains two pattern syntax.
# { } is a pattern syntax.
# 'chi:i ...' lets you select and it wants to select the output of each loop of 'for i,ch in enumerate(chars)'
# and then reverse the order of each loops result.
#
# The previous "reverse the order of each loops result" gives you an enumerable object of properties
# which is then given to the '{ }' pattern syntax to programmatically generate the object.
# This object is just 1 or many 'property'-'value' relationships and thus it is described as a dictionary.

# Maps an integer ID back to a string character: e.g., {0: 'a', 1: 'b', 2: 'c'}
int_to_char = { i:ch for i,ch in enumerate(chars) }
# ^
#
# (SPECULATION...
#  this is all my own speculation on what I'm reading here, and I have no idea what this actually means)
#
# This is the same as the previous statement but selecting 'i:ch' instead of 'ch:i'.
#
# Althought I was wondering why it had to do:
# { i:ch for i,ch in enumerate(chars) }
# Versus:
# { i,ch in enumerate(chars) }
#
# and I see now that 'i:ch' is a colon between the 'i' and the 'ch'. Whereas the latter is 'i,ch' so a comma between the 'i' and the 'ch'.
# 
# I wonder whether you could re-use 'for i,ch in enumerate(chars)'
#
# enumerable = for i,ch in enumerate(chars)
# char_to_int = { ch:i enumerable }
# int_to_char = { i:ch enumerable }

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

# ==========================================
# 9. THE BATCH GENERATOR
# ==========================================
# Number of independent text sequences to process in parallel
batch_size = 4 

def get_batch(split):
    # Select whether we are pulling from the training or validation set
    data = train_data if split == 'train' else val_data
    
    # Generate random starting points in the data array
    # We subtract block_size so we don't accidentally overflow past the end of the text
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Stack the random inputs and targets into matrices
    x_batch = torch.stack([data[i:i+block_size] for i in ix])
    y_batch = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x_batch, y_batch

# Grab a sample batch from our training split to test it
xb, yb = get_batch('train')

print("-" * 30)
print(f"Inputs Batch Shape (xb):  {xb.shape}  -> (batch_size, block_size)")
print(f"Targets Batch Shape (yb): {yb.shape}")
print("-" * 30)
print("Here is what the input batch matrix looks like inside:")
print(xb)




