import subprocess
import sys
from scipy.stats import binom_test

FIPS_RNG_MAX_LONG_RUN = 25

def fips_run_rng_test(binary_sequence):
    long_run_test_failed = False
    zeros_run = 0
    ones_run = 0

    # Iterate through the binary sequence
    for bit in binary_sequence:
        # Check for consecutive zeros
        if bit == '0':
            zeros_run += 1
            ones_run = 0
        # Check for consecutive ones
        else:
            ones_run += 1
            zeros_run = 0

        # Check if the long run threshold is exceeded
        if zeros_run > FIPS_RNG_MAX_LONG_RUN or ones_run > FIPS_RNG_MAX_LONG_RUN:
            long_run_test_failed = True
            break

    # Set the FIPS_RNG_LONGRUN flag if the long run test failed
    if long_run_test_failed:
        return 1
    else:
        return 0

def apply_longrun_test_on_blocks(stream, block_size):
    passed_blocks = 0
    total_blocks = 0

    while True:
        block = stream.read(block_size)
        if not block:
            break
        # Apply long run test on the block
        result = fips_run_rng_test(block)
        total_blocks += 1
        if result == 0:
            passed_blocks += 1

    return passed_blocks, total_blocks

if __name__ == "__main__":
    # Check if the filename is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py filename")
        sys.exit(1)
        
    filename = sys.argv[1]

    # Run the command to read binary data from the file
    command = f"xxd -b {filename} | grep -oE '[01]{{8}}' | tr -d '\n'"
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as process:
        passed_blocks, total_blocks = apply_longrun_test_on_blocks(process.stdout, block_size=20000)
        print("Passed Blocks:", passed_blocks)
        print("Total Blocks:", total_blocks)

    failed_blocks = total_blocks - passed_blocks

    # Given parameters
    max_run_length = FIPS_RNG_MAX_LONG_RUN 
    block_size = 20000  # Assuming a block size of 20000 bits

    # Step 1: Calculate the probability of success (p) and failure (q)
    p = (1 - (1 / (2**(max_run_length + 1)))) ** block_size
    q = 1 - p

    # Step 2: Calculate the expected number of successful and unsuccessful blocks
    expected_successes = total_blocks * p
    expected_failures = total_blocks * q

    # Step 3: Use the binomial test to calculate the p-value for unsuccessful blocks
    p_value = binom_test(failed_blocks, total_blocks, q, alternative='greater')
    print("successful blocks:", passed_blocks)
    print("unsuccessful blocks:", (total_blocks-passed_blocks))
    print("P-value:", p_value)
