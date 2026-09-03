# Google AI:
# This single Python file will serve as the core workspace where you will load your text data, build the neural network layers, and run the training loop.

# 1. READ THE DATA
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 2. CREATE THE VOCABULARY
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Unique characters (Vocabulary Size): {vocab_size}")

# 3. BUILD THE LOOKUP DICTIONARIES
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }

# 4. DEFINE ENCODE & DECODE FUNCTIONS
def encode(string_input):
    return [char_to_int[c] for c in string_input]
def decode(list_input):
    return "".join([int_to_char[i] for i in list_input])

# 6. CONVERT THE ENTIRE DATASET INTO TENSORS
import torch

all_encoded_data = encode(text)

data_tensor = torch.tensor(all_encoded_data, dtype=torch.long)

print("-" * 30)
print(f"Dataset shape: {data_tensor.shape}") # console output: Dataset shape: torch.Size([219])
print(f"First 10 tokens as a Tensor: {data_tensor[:10]}") # console output: First 10 tokens as a Tensor: tensor([ 7, 14, 20, 14, 27, 14,  1, 30, 17, 10])

# 7. SPLIT INTO TRAINING & VALIDATION SETS
n = int(0.9 * len(data_tensor)) # Calculate the index where the 90% mark sits
train_data = data_tensor[:n]    # The first 90% of the numerical array is for training
val_data = data_tensor[n:]      # The remaining 10% is for validating the AI's performance

print("-" * 30)
print(f"Training data size:   {len(train_data)} tokens") # Training data size:   197 tokens
print(f"Validation data size: {len(val_data)} tokens") # Validation data size: 22 tokens

# 8. CREATE INPUT AND TARGET CHUNKS (X & Y)
block_size = 8                 # Look at an 8-character window at a time
x = train_data[:block_size]    # Grab the first 8 characters as our input (X)
y = train_data[1:block_size+1] # Grab the next 8 characters, shifted forward by exactly 1 slot (Y)

print("-" * 30)
print(f"Input Tensor (x):  {x}") # Input Tensor (x):  tensor([ 7, 14, 20, 14, 27, 14,  1, 30])
print(f"Target Tensor (y): {y}") # Target Tensor (y): tensor([14, 20, 14, 27, 14,  1, 30, 17])
print("-" * 30)

# Visual explanation of what the AI sees character-by-character
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context.tolist()}, the target next character is: {target}")

# 9. THE BATCH GENERATOR
batch_size = 4 # Number of independent text sequences to process in parallel

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
print(f"Inputs Batch Shape (xb):  {xb.shape}  -> (batch_size, block_size)") # console output: Inputs Batch Shape (xb):  torch.Size([4, 8])  -> (batch_size, block_size)
print(f"Targets Batch Shape (yb): {yb.shape}") # console output: Targets Batch Shape (yb): torch.Size([4, 8])
print("-" * 30)
print("Here is what the input batch matrix looks like inside:")
print(xb) # console output: tensor([[20, 14, 27, 14,  1, 30, 17, 10],
          #                         [20, 14, 10, 26, 27,  1, 10,  1],
          #                         [ 5,  1, 23, 25,  1, 10,  1, 24],
          #                         [27,  1,  3, 10, 27,  1, 20, 14]])

import torch.nn as nn
from torch.nn import functional as F

# 10. THE NEURAL NETWORK BLUEPRINT

# This is the point where everything turns to confusion for me

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # The Embedding Layer lives here! 
        # It's a massive table of size (vocab_size x vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        # idx and targets are both (batch_size, block_size) matrices of integers
        
        # Every integer in 'idx' looks up a corresponding row of vectors in the table
        # logits represents the raw prediction scores for the next character
        logits = self.token_embedding_table(idx) # Output shape: (batch_size, block_size, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # PyTorch's cross_entropy expects data to be shaped a specific way,
            # so we flatten our matrices into vectors to calculate the error (loss)
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            
            # Calculate how wrong the AI's guesses were compared to the actual targets
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss

    # ==========================================
    # 11. THE GENERATION ENGINE
    # ==========================================
    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) matrix of character integers in the current context
        for _ in range(max_new_tokens):
            # 1. Get the predictions for the next characters
            logits, loss = self.forward(idx)
            
            # 2. Focus ONLY on the very last time step (-1) to predict the future
            logits = logits[:, -1, :] # Becomes shape (B, C)
            
            # 3. Apply a Softmax mathematical formula to convert raw scores into percentages
            probs = F.softmax(logits, dim=-1) # Shape (B, C)
            
            # 4. Sample randomly from that percentage distribution to get the next character ID
            idx_next = torch.multinomial(probs, num_samples=1) # Shape (B, 1)
            
            # 5. Glue the new character ID onto the end of our ongoing running sequence
            idx = torch.cat((idx, idx_next), dim=1) # Shape (B, T+1)
            
        return idx

# Instantiate the model using our vocabulary size
model = BigramLanguageModel(vocab_size)

# Test it by feeding it our sample batch from earlier!
logits, loss = model(xb, yb)
print("-" * 30)
print(f"Model successfully built!")
print(f"Predictions matrix shape (logits): {logits.shape}")
print(f"Initial raw error score (loss):     {loss.item():.4f}")

# Create a 1x1 matrix containing just the integer 0 (assuming 0 is a newline or space token)
# This acts as our "seed" or starting point for the text generation
context = torch.zeros((1, 1), dtype=torch.long)

# Ask the model to generate 100 brand new tokens, decode the array back to text, and print it
print("-" * 30)
# We call model.generate, extract the 0th batch row, convert it to a standard Python list, and decode it
print("UNTRAINED MODEL OUTPUT:")
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
print("-" * 30)

# ==========================================
# 12. THE TRAINING ENGINE (THE OPTIMIZER)
# ==========================================
# Create a PyTorch Optimizer (AdamW is the standard industry-strength choice)
# It takes our model's weights and a learning rate (how fast it should adjust things)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Training the model... please wait a few seconds...")
print("-" * 30)

# We will run the training loop for 10,000 rounds (steps)
for steps in range(10000):
    
    # 1. Grab a fresh, random batch of training data (Inputs and Targets)
    xb, yb = get_batch('train')
    
    # 2. Feed the batch to the model to get the predictions and the error score
    logits, loss = model(xb, yb)
    
    # 3. Wipe out the memory of old calculations from the previous step
    optimizer.zero_grad(set_to_none=True)
    
    # 4. Perform "Backpropagation" - calculate how much each weight contributed to the error
    loss.backward()
    
    # 5. Tell the optimizer to tweak the weights based on the calculation above
    optimizer.step()
    
    # Every 2,000 steps, let's print out our progress to see the error drop
    if steps % 2000 == 0:
        print(f"Step {steps:5d} | Current Loss: {loss.item():.4f}")

print(f"Step 10000 | Final Loss: {loss.item():.4f}")
print("-" * 30)

# ==========================================
# 13. TEST THE TRAINED MODEL
# ==========================================
print("TRAINED MODEL OUTPUT:")
# Notice the [0] added right after generate(...)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
print("-" * 30)










# 5. TEST IT OUT
def testItOut():
    sample_phrase = "hello"
    encoded_phrase = encode(sample_phrase)

    print("-" * 30)
    print(f"Original Text: {sample_phrase}") # console output: Original Text: hello
    print(f"Encoded Numbers: {encoded_phrase}") # console output: Encoded Numbers: [17, 14, 20, 20, 23]
    print(f"Decoded Back:   {decode(encoded_phrase)}") # console output: Decoded Back:   hello
