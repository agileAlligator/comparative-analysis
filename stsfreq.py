import subprocess
import sys
import sts

def get_binary_representation(file_path):
    # Run the command to get the binary representation
    command = f"xxd -b {file_path} | grep -oE '[01]{{8}}' | tr -d '\n'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Convert the string of binary digits to a list of integers
        binary_representation = result.stdout.strip()
        binary_list = [int(bit) for bit in binary_representation]
        return binary_list
    else:
        print("Error:", result.stderr)
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    RNG_output = get_binary_representation(file_path)
    
    if RNG_output is not None:
        results = {}
        results["Frequency within Block"] = sts.block_frequency(RNG_output, 1000)
        print("Results:")
        print(results)
    else:
        print("Failed to obtain binary representation from the file.")
