import subprocess
import sys
import re

def run_rab(filename):
    try:
        result = subprocess.run(["./testu01/rab", filename], check=True, capture_output=True, text=True)
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if 'HammingIndep, L = 16' in line:
                print(line)
            if 'Run of bits' in line:
                print(line)
            if 'HammingCorr, L = 32' in line:
                print(line)
            if 'MultinomialBitsOver' in line:
                print(line)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

def run_stsfreq(filename):
    try:
        result = subprocess.run(["python3","./sts-pylib/stsfreq.py", filename], check=True, capture_output=True, text=True)
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            # Modify this condition to extract relevant information from the output
            if 'p_value' in line:
                print("STS frequencyi within block",line)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python helper.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    run_rab(filename)
    run_stsfreq(filename)
