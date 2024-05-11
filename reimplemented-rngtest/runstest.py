import subprocess
import sys
from scipy.stats import binom_test

def runs_test(binary_sequence):
    runs_counts = [0] * 12

    current_run_length = 1
    current_bit = binary_sequence[0]

    for bit in binary_sequence[1:]:
        if bit == current_bit:
            current_run_length += 1
        else:
            if current_run_length <= 6:
                runs_counts[current_run_length - 1 + (6 * current_bit)] += 1
            else:
                runs_counts[5 + (6 * current_bit)] += 1
            current_run_length = 1
            current_bit = bit
    
    # Add the last run
    if current_run_length <= 6:
        runs_counts[current_run_length - 1 + (6 * current_bit)] += 1
    else:
        runs_counts[5 + (6 * current_bit)] += 1

    # Check if any run length is outside the acceptable range
    if any(count < lower_bound or count > upper_bound
           for count, lower_bound, upper_bound in zip(runs_counts,
                                                       [2315, 1114, 527, 240, 103, 103, 2315, 1114, 527, 240, 103, 103],
                                                       [2685, 1386, 723, 384, 209, 209, 2685, 1386, 723, 384, 209, 209])):
        return False
    else:
        return True

def apply_runs_test_on_blocks(binary_stream, block_size):
    passed_blocks = 0
    total_blocks = 0

    while True:
        block = binary_stream.read(block_size)
        if not block:
            break

        binary_sequence = [int(bit) for bit in block]

        if runs_test(binary_sequence):
            passed_blocks += 1
        total_blocks += 1

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
        # Apply the runs test on the data stream
        passed_blocks, total_blocks = apply_runs_test_on_blocks(process.stdout, block_size=20000)
        print("Passed blocks:", passed_blocks)
        print("Total blocks:", total_blocks)

        p_value = binom_test((total_blocks-passed_blocks), total_blocks, p=0.005, alternative='greater')
        print("P-value:", p_value)
