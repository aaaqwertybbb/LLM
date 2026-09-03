# Google AI:
# This single Python file will serve as the core workspace where you will load your text data, build the neural network layers, and run the training loop.

# 1. Open the file in read-only mode ('r') using UTF-8 encoding
with open("input.txt", "r", encoding="utf-8") as file:
    # 2. Read the entire text contents into a variable
    text = file.read()

# 3. Print the total number of characters in your dataset
print(f"Total characters: {len(text)}")

# 4. Print a divider line to make it look clean
print("-" * 20)

# 5. Print just the first 250 characters to test it out
print(text[:250])
