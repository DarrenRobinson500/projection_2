import os

def is_directory_readable_writable(directory_path):
    try:
        os.access(directory_path, os.R_OK | os.W_OK)
        return True
    except PermissionError:
        return False

def is_file_readable_writable(file_path):
    try:
        with open(file_path, 'r') as file:
            # File is not locked
            print("File is not locked.")
            pass
    except IOError:
        # File is locked
        print("File is locked.")


# Usage
# directory_path = "/path/to/your/directory"

file_path = f"files\\start\30_June_2023.csv"
directory_path = f"files\\start"

if is_directory_readable_writable(directory_path):
    print("File is readable and writable.")
else:
    print("File does not have read and/or write access.")

