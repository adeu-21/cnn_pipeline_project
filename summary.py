# Extract the last two lines from output_nonpara.txt
with open("./output_nonpara.txt", "r") as file:  # Update path if needed
    lines = file.readlines()
last_two_lines = lines[-2:]

# Write non-parallel results to result.txt
result_file = open("./result.txt", "w")
result_file.writelines("For non-parallel part:\n")
result_file.writelines(last_two_lines)

# Extract the last two lines from output_para.txt
with open("./output_para.txt", "r") as file:
    lines = file.readlines()
last_two_lines = lines[-2:]

# Write parallel results to result.txt
result_file.writelines("\n\nFor parallel part:\n")
result_file.writelines(last_two_lines)

result_file.close()

# Read the result file and calculate speedup
file_path = "./result.txt"

# Read the file to extract training times
with open(file_path, "r") as file:
    lines = file.readlines()

# Parse training times for non-parallel and parallel parts
non_parallel_time = float(lines[1].split(" ")[3])
parallel_time = float(lines[6].split(" ")[3])

# Calculate speedup
speedup = non_parallel_time / parallel_time if parallel_time else None

# Append the speedup to the result file
if speedup is not None:
    with open("./result.txt", "a") as file:
        file.write(f"\n\nSpeedup: {speedup:.2f}\n")
