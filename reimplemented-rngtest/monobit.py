import numpy as np
from scipy.stats import binom
from scipy import stats
import collections
from scipy.stats import binom_test
import sys
import subprocess

def monobit_test(data):
    # Split the data into blocks of 20000 bits
    blocks = [data[i:i+20000] for i in range(0, len(data), 20000)]
    passed_blocks = 0
    failed_blocks = 0
    ones_counts = []
    
    for block in blocks:
        # Number of ones in the current block
        ones_count = np.sum(block)
        ones_counts.append(ones_count)
        
        # Check if the number of ones falls within the specified range
        if ones_count >= 9725 and ones_count <= 10275:
            passed_blocks += 1
        else:
            failed_blocks += 1
    
    # Calculate the overall proportion of blocks passing the Monobit test
    proportion_passed_blocks = passed_blocks / len(blocks)
    p_value = binom_test(failed_blocks, (passed_blocks + failed_blocks), p=0.005, alternative='greater')
    return passed_blocks, failed_blocks, proportion_passed_blocks, p_value

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

    passed_blocks, failed_blocks, proportion_passed_blocks, p_value = monobit_test(binary_array)
    print("Number of blocks passing the Monobit test:", passed_blocks)
    print("Number of blocks failing the Monobit test:", failed_blocks)
    print("Proportion of blocks passing the Monobit test:", proportion_passed_blocks)
    print("P-value:", p_value)
