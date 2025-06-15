import torch
import subprocess

# Check if GPU (CUDA) is available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Display system specifications
def print_system_specs():
    # Show CPU details (Linux/macOS)
    print("\nCPU Info:")
    try:
        cpu_info = subprocess.check_output(
            "lscpu | grep 'Model name\|Socket(s)\|Core(s) per socket\|Thread(s) per core\|CPU MHz\|CPU(s)'",
            shell=True,
        ).decode()
        print(cpu_info)
    except subprocess.CalledProcessError:
        print("Unable to fetch CPU information")

    # Show GPU details if available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\nNumber of GPUs: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            print(f"GPU {i}: {gpu_name}")
            print(f"  Total GPU Memory: {gpu_mem:.2f} GB")
    else:
        print("No GPU available.")

    # Show RAM details (Linux/macOS)
    print("\nRAM Info:")
    try:
        ram_info = subprocess.check_output(
            'free -h --si | awk \'/^Mem:/{print "Total RAM: " $2 ", Used RAM: " $3 ", Free RAM: " $4}\'',
            shell=True,
        ).decode()
        print(ram_info)
    except subprocess.CalledProcessError:
        print("Unable to fetch RAM information")


print_system_specs()
