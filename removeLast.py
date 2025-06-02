import numpy as np

# Load the data from the .npy file
data = np.load("experiment_data.npy")
print(len(data))
# Define the number of entries to remove
x = int(input("How many to delete of the end?"))  # Change this value as needed

# Remove the last x entries
if x > 0:
    data = data[:-x]

print(len(data))
if input("Are you sure?") == "y":
    print("Removing last x entries")
    # Save the modified data back to the .npy file
    np.save("experiment_data.npy", data)