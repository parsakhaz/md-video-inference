# To run: modal run example-script.py

import modal

app = modal.App("example-get-started")

# Define a function that will run on a remote worker
@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2

# Define the entrypoint for the app
@app.local_entrypoint()
def main():
    print("the square is", square.remote(42))

# Congratulations, you successfully executed a function on a remote worker!