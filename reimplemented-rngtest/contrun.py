import subprocess
import sys
from scipy.stats import binom_test
def cont_run_blocks(u, n, m, bits):
    a = n // m
    passed_blocks = 0
    failed_blocks = 0
    for i in range(a):
        b = m // bits
        r = min(bits * b, n - i * m)  # Ensure we don't go beyond the end of the data
        v = [u[start:start+bits] for start in range(i * m, i * m + r, bits)]
        failed = False
        for j in range(b - 1):
            if v[j] == v[j+1]:
                failed = True
                break
        if failed:
            failed_blocks += 1
        else:
            passed_blocks += 1
    return passed_blocks, failed_blocks

if __name__ == "__main__":
    # Check if the filename is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py filename")
        sys.exit(1)
        
    file_path = sys.argv[1]

    # Execute the command and capture the output
    command = f"xxd -b {file_path} | grep -oE '[01]{{8}}' | tr -d '\n'"
    output = subprocess.check_output(command, shell=True, text=True)

    # Convert the output into a list of integers
    data = [int(bit) for bit in output]

    # Parameters for cont_run_blocks function
    u = data
    n = len(data)
    m = 20000  # Block size
    bits = 32   # Number of bits per unit (assuming we're checking for consecutive 32-bit patterns)

    # Get the number of passed and failed blocks
    passed_blocks, failed_blocks = cont_run_blocks(u, n, m, bits)
    print("Number of passed blocks:", passed_blocks)
    print("Number of failed blocks:", failed_blocks)
    p_value = binom_test(failed_blocks,(passed_blocks+failed_blocks),p=0.005, alternative='greater')
    print("P-value: ",p_value)
                              
