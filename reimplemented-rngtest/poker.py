import numpy as np
from scipy.stats import binom_test
import subprocess
import sys

def poker_test(data):
    # Initialize an array to store the frequencies of each possible outcome
    poker = [0] * 16
    
    # Convert the data into blocks of 4 bits each and calculate the frequencies
    for i in range(0, len(data), 4):
        # Extract 4 bits from the data and convert it to an integer
        block = int(data[i:i+4], 2)
        # Increment the corresponding frequency count
        poker[block] += 1
    
    # Calculate the sum of squared frequencies
    j = sum(freq ** 2 for freq in poker)
    
    # Define the threshold values
    lower_threshold = 1563176
    upper_threshold = 1576928
    
    # Check if the sum falls outside the threshold range
    if j < lower_threshold or j > upper_threshold:
        return False
    else:
        return True

def calculate_p_value(passed_blocks, total_blocks):
    # Calculate the overall p-value
    p_value = binom_test((total_blocks-passed_blocks), total_blocks, p=0.005,alternative='greater')
    return p_value

def apply_poker_test_on_blocks(data, block_size):
    # Convert the binary data array to a string
    data_str = ''.join(str(bit) for bit in data)
    
    # Split the data into blocks of block_size bits
    blocks = [data_str[i:i+block_size] for i in range(0, len(data_str), block_size)]
    
    passed_blocks = 0
    
    # Apply the poker test on each block
    for block in blocks:
        if poker_test(block):
            passed_blocks += 1
    
    return passed_blocks, len(blocks)

if __name__ == "__main__":
    # Check if the filename is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py filename")
        sys.exit(1)
        
    filename = sys.argv[1]
    
    # Run the command to read binary data from the file
    command = f"xxd -b {filename} | grep -oE '[01]{{8}}' | tr -d '\n'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Check if the command ran successfully
    if result.returncode == 0:
        # Assign the output to a variable
        binary_representation = result.stdout.strip()
        binary_array = np.array([int(bit) for bit in binary_representation])
    else:
        print("Error:", result.stderr)
        sys.exit(1)

    # Define the block size
    block_size = 20000

    # Apply the poker test on the blocks
    passed_blocks, total_blocks = apply_poker_test_on_blocks(binary_array, block_size)

    # Calculate the p-value
    p_value = calculate_p_value(passed_blocks, total_blocks)

    print("Number of blocks passing the Poker test:", passed_blocks)
    print("Number of total blocks:", total_blocks)
    print("P-value:", p_value)
