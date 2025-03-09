import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display, HTML

# Try to import GPU-specific libraries with error handling
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("PyTorch not available. Some tests will be skipped.")

try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cupy_available = False
    print("CuPy not available. Some tests will be skipped.")

# Helper function to format time
def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds*1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds*1000:.2f} ms"
    else:
        return f"{seconds:.4f} s"

# Function to get GPU information
def get_gpu_info():
    info = {"GPU Available": False, "Details": {}}
    
    if torch_available and torch.cuda.is_available():
        info["GPU Available"] = True
        info["Library"] = "PyTorch"
        info["Details"]["Device Count"] = torch.cuda.device_count()
        info["Details"]["Current Device"] = torch.cuda.current_device()
        info["Details"]["Device Name"] = torch.cuda.get_device_name(0)
        try:
            info["Details"]["Memory Allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
            info["Details"]["Memory Reserved"] = f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
        except:
            info["Details"]["Memory Stats"] = "Not available"
    
    return info

# Display GPU information
def display_gpu_info():
    gpu_info = get_gpu_info()
    if gpu_info["GPU Available"]:
        print(f"🎉 GPU detected: {gpu_info['Details']['Device Name']}")
        
        # Create a styled HTML table
        html = """
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h3 style="color: #2c3e50; margin-top: 0;">NVIDIA GPU Information</h3>
            <table style="width: 100%; border-collapse: collapse;">
        """
        
        for key, value in gpu_info["Details"].items():
            html += f"""
            <tr style="border-bottom: 1px solid #ddd;">
                <td style="padding: 8px; color: #34495e; font-weight: bold;">{key}</td>
                <td style="padding: 8px;">{value}</td>
            </tr>
            """
        
        html += """
            </table>
        </div>
        """
        
        display(HTML(html))
    else:
        print("❌ No NVIDIA GPU detected or CUDA is not available")

# Matrix multiplication test
def test_matrix_multiplication(sizes=[1000, 2000, 4000], repeat=3):
    results = []
    
    print("🧪 Running matrix multiplication tests...")
    
    for size in sizes:
        print(f"  Testing size {size}x{size}...")
        
        # CPU test
        a_cpu = np.random.random((size, size)).astype(np.float32)
        b_cpu = np.random.random((size, size)).astype(np.float32)
        
        # Warm up
        _ = np.dot(a_cpu, b_cpu)
        
        # Actual timing
        cpu_times = []
        for i in range(repeat):
            start = time.time()
            _ = np.dot(a_cpu, b_cpu)
            end = time.time()
            cpu_times.append(end - start)
        
        cpu_time = np.mean(cpu_times)
        results.append({
            "Device": "CPU (NumPy)",
            "Matrix Size": f"{size}x{size}",
            "Time": cpu_time,
            "Time Formatted": format_time(cpu_time)
        })
        
        # PyTorch CPU test
        if torch_available:
            a_torch_cpu = torch.tensor(a_cpu)
            b_torch_cpu = torch.tensor(b_cpu)
            
            # Warm up
            _ = torch.matmul(a_torch_cpu, b_torch_cpu)
            
            # Actual timing
            torch_cpu_times = []
            for i in range(repeat):
                start = time.time()
                _ = torch.matmul(a_torch_cpu, b_torch_cpu)
                end = time.time()
                torch_cpu_times.append(end - start)
            
            torch_cpu_time = np.mean(torch_cpu_times)
            results.append({
                "Device": "CPU (PyTorch)",
                "Matrix Size": f"{size}x{size}",
                "Time": torch_cpu_time,
                "Time Formatted": format_time(torch_cpu_time)
            })
        
        # PyTorch GPU test
        if torch_available and torch.cuda.is_available():
            a_torch_gpu = a_torch_cpu.cuda()
            b_torch_gpu = b_torch_cpu.cuda()
            
            # Warm up
            _ = torch.matmul(a_torch_gpu, b_torch_gpu)
            torch.cuda.synchronize()
            
            # Actual timing
            torch_gpu_times = []
            for i in range(repeat):
                start = time.time()
                _ = torch.matmul(a_torch_gpu, b_torch_gpu)
                torch.cuda.synchronize()
                end = time.time()
                torch_gpu_times.append(end - start)
            
            torch_gpu_time = np.mean(torch_gpu_times)
            results.append({
                "Device": "GPU (PyTorch)",
                "Matrix Size": f"{size}x{size}",
                "Time": torch_gpu_time,
                "Time Formatted": format_time(torch_gpu_time)
            })
            
            # Calculate speedup
            speedup = cpu_time / torch_gpu_time
            print(f"  ⚡ GPU is {speedup:.2f}x faster than CPU for {size}x{size} matrices")
        
        # CuPy GPU test
        if cupy_available:
            a_cupy = cp.array(a_cpu)
            b_cupy = cp.array(b_cpu)
            
            # Warm up
            _ = cp.dot(a_cupy, b_cupy)
            cp.cuda.Stream.null.synchronize()
            
            # Actual timing
            cupy_times = []
            for i in range(repeat):
                start = time.time()
                _ = cp.dot(a_cupy, b_cupy)
                cp.cuda.Stream.null.synchronize()
                end = time.time()
                cupy_times.append(end - start)
            
            cupy_time = np.mean(cupy_times)
            results.append({
                "Device": "GPU (CuPy)",
                "Matrix Size": f"{size}x{size}",
                "Time": cupy_time,
                "Time Formatted": format_time(cupy_time)
            })
    
    return pd.DataFrame(results)

# Function to test FFT performance
def test_fft(sizes=[2**20, 2**22, 2**24], repeat=3):
    results = []
    
    print("🧪 Running FFT tests...")
    
    for size in sizes:
        print(f"  Testing size {size}...")
        
        # CPU test
        a_cpu = np.random.random(size).astype(np.complex64)
        
        # Warm up
        _ = np.fft.fft(a_cpu)
        
        # Actual timing
        cpu_times = []
        for i in range(repeat):
            start = time.time()
            _ = np.fft.fft(a_cpu)
            end = time.time()
            cpu_times.append(end - start)
        
        cpu_time = np.mean(cpu_times)
        results.append({
            "Device": "CPU (NumPy)",
            "FFT Size": f"{size}",
            "Time": cpu_time,
            "Time Formatted": format_time(cpu_time)
        })
        
        # CuPy GPU test
        if cupy_available:
            a_cupy = cp.array(a_cpu)
            
            # Warm up
            _ = cp.fft.fft(a_cupy)
            cp.cuda.Stream.null.synchronize()
            
            # Actual timing
            cupy_times = []
            for i in range(repeat):
                start = time.time()
                _ = cp.fft.fft(a_cupy)
                cp.cuda.Stream.null.synchronize()
                end = time.time()
                cupy_times.append(end - start)
            
            cupy_time = np.mean(cupy_times)
            results.append({
                "Device": "GPU (CuPy)",
                "FFT Size": f"{size}",
                "Time": cupy_time,
                "Time Formatted": format_time(cupy_time)
            })
            
            # Calculate speedup
            speedup = cpu_time / cupy_time
            print(f"  ⚡ GPU is {speedup:.2f}x faster than CPU for FFT size {size}")
    
    return pd.DataFrame(results)

# Function to test neural network operations
def test_neural_net_ops(batch_sizes=[64, 128, 256, 512], repeat=3):
    if not (torch_available and torch.cuda.is_available()):
        print("❌ PyTorch with CUDA is required for neural network tests")
        return pd.DataFrame()
    
    results = []
    print("🧪 Running neural network operation tests...")
    
    # Define a simple CNN
    class SimpleCNN(torch.nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = torch.nn.MaxPool2d(2, 2)
            self.fc1 = torch.nn.Linear(128 * 56 * 56, 512)
            self.fc2 = torch.nn.Linear(512, 10)
            self.relu = torch.nn.ReLU()
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 128 * 56 * 56)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Test forward and backward pass
    for batch_size in batch_sizes:
        print(f"  Testing batch size {batch_size}...")
        
        # CPU model and data
        model_cpu = SimpleCNN()
        inputs_cpu = torch.randn(batch_size, 3, 224, 224)
        
        # Warm up
        _ = model_cpu(inputs_cpu)
        
        # Forward pass timing (CPU)
        cpu_forward_times = []
        for i in range(repeat):
            start = time.time()
            outputs_cpu = model_cpu(inputs_cpu)
            end = time.time()
            cpu_forward_times.append(end - start)
        
        cpu_forward_time = np.mean(cpu_forward_times)
        results.append({
            "Device": "CPU",
            "Operation": "Forward Pass",
            "Batch Size": batch_size,
            "Time": cpu_forward_time,
            "Time Formatted": format_time(cpu_forward_time)
        })
        
        # Backward pass timing (CPU)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
        targets_cpu = torch.randint(0, 10, (batch_size,))
        
        # Warm up
        loss_cpu = criterion(outputs_cpu, targets_cpu)
        loss_cpu.backward()
        
        cpu_backward_times = []
        for i in range(repeat):
            optimizer_cpu.zero_grad()
            outputs_cpu = model_cpu(inputs_cpu)
            loss_cpu = criterion(outputs_cpu, targets_cpu)
            
            start = time.time()
            loss_cpu.backward()
            end = time.time()
            
            cpu_backward_times.append(end - start)
        
        cpu_backward_time = np.mean(cpu_backward_times)
        results.append({
            "Device": "CPU",
            "Operation": "Backward Pass",
            "Batch Size": batch_size,
            "Time": cpu_backward_time,
            "Time Formatted": format_time(cpu_backward_time)
        })
        
        # GPU model and data
        model_gpu = SimpleCNN().cuda()
        inputs_gpu = inputs_cpu.cuda()
        targets_gpu = targets_cpu.cuda()
        optimizer_gpu = torch.optim.SGD(model_gpu.parameters(), lr=0.01)
        
        # Warm up
        _ = model_gpu(inputs_gpu)
        torch.cuda.synchronize()
        
        # Forward pass timing (GPU)
        gpu_forward_times = []
        for i in range(repeat):
            torch.cuda.synchronize()
            start = time.time()
            outputs_gpu = model_gpu(inputs_gpu)
            torch.cuda.synchronize()
            end = time.time()
            gpu_forward_times.append(end - start)
        
        gpu_forward_time = np.mean(gpu_forward_times)
        results.append({
            "Device": "GPU",
            "Operation": "Forward Pass",
            "Batch Size": batch_size,
            "Time": gpu_forward_time,
            "Time Formatted": format_time(gpu_forward_time)
        })
        
        # Calculate forward pass speedup
        forward_speedup = cpu_forward_time / gpu_forward_time
        print(f"  ⚡ GPU is {forward_speedup:.2f}x faster than CPU for forward pass with batch size {batch_size}")
        
        # Backward pass timing (GPU)
        loss_gpu = criterion(outputs_gpu, targets_gpu)
        loss_gpu.backward()  # Warm up
        torch.cuda.synchronize()
        
        gpu_backward_times = []
        for i in range(repeat):
            optimizer_gpu.zero_grad()
            outputs_gpu = model_gpu(inputs_gpu)
            loss_gpu = criterion(outputs_gpu, targets_gpu)
            
            torch.cuda.synchronize()
            start = time.time()
            loss_gpu.backward()
            torch.cuda.synchronize()
            end = time.time()
            
            gpu_backward_times.append(end - start)
        
        gpu_backward_time = np.mean(gpu_backward_times)
        results.append({
            "Device": "GPU",
            "Operation": "Backward Pass",
            "Batch Size": batch_size,
            "Time": gpu_backward_time,
            "Time Formatted": format_time(gpu_backward_time)
        })
        
        # Calculate backward pass speedup
        backward_speedup = cpu_backward_time / gpu_backward_time
        print(f"  ⚡ GPU is {backward_speedup:.2f}x faster than CPU for backward pass with batch size {batch_size}")
    
    return pd.DataFrame(results)

# Visualize results
def plot_results(df, title, x_col, y_col="Time", hue_col="Device", log_scale=True):
    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(x=x_col, y=y_col, hue=hue_col, data=df)
    
    plt.title(title, fontsize=16)
    if log_scale:
        plt.yscale('log')
        plt.ylabel("Time (seconds) - Log Scale")
    else:
        plt.ylabel("Time (seconds)")
    
    # Add value labels on the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    plt.tight_layout()
    plt.legend(title=hue_col)
    return plt

# Compile all tests and visualizations
def run_all_tests():
    display_gpu_info()
    
    print("\n" + "="*50)
    print("NVIDIA GPU Performance Tests")
    print("="*50 + "\n")
    
    # Matrix multiplication test
    matmul_results = test_matrix_multiplication(sizes=[1000, 2000, 4000], repeat=3)
    
    if not matmul_results.empty:
        print("\n📊 Matrix Multiplication Results:")
        display(matmul_results)
        
        plot = plot_results(matmul_results, 
                           "Matrix Multiplication Performance Comparison", 
                           "Matrix Size", 
                           y_col="Time",
                           hue_col="Device")
        plot.show()
    
    # FFT test
    fft_results = test_fft(sizes=[2**20, 2**22, 2**24], repeat=3)
    
    if not fft_results.empty:
        print("\n📊 FFT Results:")
        display(fft_results)
        
        plot = plot_results(fft_results, 
                           "FFT Performance Comparison", 
                           "FFT Size", 
                           y_col="Time",
                           hue_col="Device")
        plot.show()
    
    # Neural network test
    nn_results = test_neural_net_ops(batch_sizes=[64, 128, 256], repeat=3)
    
    if not nn_results.empty:
        print("\n📊 Neural Network Operation Results:")
        display(nn_results)
        
        # Plot forward pass results
        forward_results = nn_results[nn_results["Operation"] == "Forward Pass"]
        plot = plot_results(forward_results, 
                           "Neural Network Forward Pass Performance", 
                           "Batch Size", 
                           y_col="Time",
                           hue_col="Device")
        plot.show()
        
        # Plot backward pass results
        backward_results = nn_results[nn_results["Operation"] == "Backward Pass"]
        plot = plot_results(backward_results, 
                           "Neural Network Backward Pass Performance", 
                           "Batch Size", 
                           y_col="Time",
                           hue_col="Device")
        plot.show()
    
    # Calculate and display overall speedup
    if torch_available and torch.cuda.is_available():
        print("\n🚀 Overall Performance Summary:")
        
        # Matrix multiplication speedup
        if not matmul_results.empty:
            gpu_times = matmul_results[matmul_results["Device"] == "GPU (PyTorch)"]["Time"].values
            cpu_times = matmul_results[matmul_results["Device"] == "CPU (NumPy)"]["Time"].values
            
            if len(gpu_times) > 0 and len(cpu_times) > 0:
                avg_speedup = np.mean(cpu_times / gpu_times)
                print(f"Average Matrix Multiplication Speedup: {avg_speedup:.2f}x")
        
        # Neural network speedup
        if not nn_results.empty:
            gpu_forward = nn_results[(nn_results["Device"] == "GPU") & 
                                     (nn_results["Operation"] == "Forward Pass")]["Time"].values
            cpu_forward = nn_results[(nn_results["Device"] == "CPU") & 
                                     (nn_results["Operation"] == "Forward Pass")]["Time"].values
            
            if len(gpu_forward) > 0 and len(cpu_forward) > 0:
                avg_forward_speedup = np.mean(cpu_forward / gpu_forward)
                print(f"Average Neural Network Forward Pass Speedup: {avg_forward_speedup:.2f}x")
            
            gpu_backward = nn_results[(nn_results["Device"] == "GPU") & 
                                      (nn_results["Operation"] == "Backward Pass")]["Time"].values
            cpu_backward = nn_results[(nn_results["Device"] == "CPU") & 
                                      (nn_results["Operation"] == "Backward Pass")]["Time"].values
            
            if len(gpu_backward) > 0 and len(cpu_backward) > 0:
                avg_backward_speedup = np.mean(cpu_backward / gpu_backward)
                print(f"Average Neural Network Backward Pass Speedup: {avg_backward_speedup:.2f}x")

# Run this in your Jupyter notebook
if __name__ == "__main__":
    run_all_tests()
