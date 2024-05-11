import tkinter as tk
from tkinter import filedialog
import subprocess

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Improved Script")
        self.geometry("900x400")

        self.file_path = ""

        self.create_widgets()

    def create_widgets(self):
        self.file_label = tk.Label(self, text="Select a file:")
        self.file_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.file_button = tk.Button(self, text="Browse", command=self.browse_file)
        self.file_button.grid(row=1, column=0, pady=10)

        self.run_scripts_button = tk.Button(self, text="Run Scripts", command=self.run_scripts)
        self.run_scripts_button.grid(row=1, column=1, pady=10)

        self.output_text_scripts = tk.Text(self, height=8, width=50)
        self.output_text_scripts.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.output_text_testu01 = tk.Text(self, height=8, width=50)
        self.output_text_testu01.grid(row=2, column=2, columnspan=2, padx=10, pady=10)

        self.run_testu01_button = tk.Button(self, text="Run Improved Script", command=self.run_testu01)
        self.run_testu01_button.grid(row=3, column=2, columnspan=2, pady=10)

    def browse_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("WebP files", "*.webp"), ("All files", "*.*")])
        if self.file_path:
            self.file_label.config(text=f"Selected file: {self.file_path}")

    def run_scripts(self):
        if self.file_path:
            scripts = ["monobit.py", "runstest.py", "contrun.py", "longruntest.py", "poker.py"]
            for script in scripts:
                output = subprocess.check_output(["python3", script, self.file_path], universal_newlines=True)
                self.output_text_scripts.insert(tk.END, f"Output of {script}:\n{output}\n")
        else:
            self.output_text_scripts.insert(tk.END, "Please select a file before running the scripts.\n")
    def run_testu01(self):
        if self.file_path:
            try:
                output = subprocess.check_output(["python3", "/home/kali/helper.py", self.file_path], universal_newlines=True)
                self.output_text_testu01.insert(tk.END, f"Output of Improved Suite:\n{output}\n")
            except subprocess.CalledProcessError as e:
                self.output_text_testu01.insert(tk.END, f"Error: {e}\n")
        else:
            self.output_text_testu01.insert(tk.END, "Please select a file before running the TestU01 script.\n")


if __name__ == "__main__":
    app = App()
    app.mainloop()
