import pynvml
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display, HTML

# Initialize NVML
def initialize_nvml():
    try:
        pynvml.nvmlInit()
        return True
    except Exception as e:
        print(f"Failed to initialize NVML: {e}")
        return False

# Get detailed GPU information using NVML
def get_nvml_gpu_info():
    if not initialize_nvml():
        return {}
    
    try:
        info = {}
        device_count = pynvml.nvmlDeviceGetCount()
        info["Device Count"] = device_count
        
        devices = []
        for i in range(device_count):
            device = {}
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Basic info
            device["Index"] = i
            device["Name"] = pynvml.nvmlDeviceGetName(handle)
            device["UUID"] = pynvml.nvmlDeviceGetUUID(handle)
            
            # Memory info
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            device["Total Memory"] = f"{memory.total / (1024**2):.2f} MB"
            device["Used Memory"] = f"{memory.used / (1024**2):.2f} MB"
            device["Free Memory"] = f"{memory.free / (1024**2):.2f} MB"
            device["Memory Utilization"] = f"{memory.used / memory.total * 100:.2f}%"
            
            # Utilization info
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            device["GPU Utilization"] = f"{utilization.gpu}%"
            device["Memory IO Utilization"] = f"{utilization.memory}%"
            
            # Temperature
            device["Temperature"] = f"{pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)}°C"
            
            # Power
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                device["Power Usage"] = f"{power_usage:.2f} W"
                device["Power Limit"] = f"{power_limit:.2f} W"
                device["Power Utilization"] = f"{power_usage / power_limit * 100:.2f}%"
            except:
                device["Power Info"] = "Not available"
            
            # Clock speeds
            try:
                device["Graphics Clock"] = f"{pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)} MHz"
                device["SM Clock"] = f"{pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)} MHz"
                device["Memory Clock"] = f"{pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)} MHz"
            except:
                device["Clock Info"] = "Not available"
            
            # PCIe info
            try:
                pcie_info = pynvml.nvmlDeviceGetPciInfo(handle)
                device["PCIe Generation"] = f"Gen {pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)}"
                device["PCIe Width"] = f"x{pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)}"
                device["PCIe Throughput"] = f"{pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / (1024**2):.2f} MB/s TX"
            except:
                device["PCIe Info"] = "Not available"
            
            # CUDA compute capability
            try:
                cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                device["CUDA Compute Capability"] = f"{cc_major}.{cc_minor}"
            except:
                device["CUDA Compute Capability"] = "Not available"
            
            devices.append(device)
        
        info["Devices"] = devices
        
        # NVML driver info
        try:
            info["Driver Version"] = pynvml.nvmlSystemGetDriverVersion()
            info["NVML Version"] = pynvml.nvmlSystemGetNVMLVersion()
        except:
            info["Version Info"] = "Not available"
        
        # Finalize NVML
        pynvml.nvmlShutdown()
        
        return info
    
    except Exception as e:
        print(f"Error getting GPU info with NVML: {e}")
        try:
            pynvml.nvmlShutdown()
        except:
            pass
        return {}

# Display NVML GPU information
def display_nvml_gpu_info():
    gpu_info = get_nvml_gpu_info()
    
    if not gpu_info or "Devices" not in gpu_info or not gpu_info["Devices"]:
        print("❌ No NVIDIA GPU detected or NVML is not available")
        return
    
    print(f"🎉 Detected {len(gpu_info['Devices'])} GPU(s)")
    
    for i, device in enumerate(gpu_info["Devices"]):
        # Create a styled HTML table for each GPU
        html = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h3 style="color: #2c3e50; margin-top: 0;">NVIDIA GPU {i}: {device.get('Name', 'Unknown')}</h3>
            <table style="width: 100%; border-collapse: collapse;">
        """
        
        # Skip 'Index' and 'Name' since they're in the header
        skip_keys = ['Index', 'Name']
        
        # Group metrics into categories
        categories = {
            "Basic Information": ["UUID", "CUDA Compute Capability"],
            "Memory": ["Total Memory", "Used Memory", "Free Memory", "Memory Utilization"],
            "Performance": ["GPU Utilization", "Memory IO Utilization", "Temperature"],
            "Power": ["Power Usage", "Power Limit", "Power Utilization"],
            "Clocks": ["Graphics Clock", "SM Clock", "Memory Clock"],
            "PCIe": ["PCIe Generation", "PCIe Width", "PCIe Throughput"]
        }
        
        # Special handling for when metrics aren't available
        fallbacks = {
            "Power": "Power Info",
            "Clocks": "Clock Info",
            "PCIe": "PCIe Info"
        }
        
        for category, metrics in categories.items():
            html += f"""
            <tr style="background-color: #e9ecef;">
                <td colspan="2" style="padding: 8px; color: #495057; font-weight: bold;">{category}</td>
            </tr>
            """
            
            # Check if any metrics in this category are available
            metrics_available = any(metric in device for metric in metrics)
            fallback = fallbacks.get(category)
            
            if metrics_available:
                for metric in metrics:
                    if metric in device:
                        html += f"""
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 8px; color: #34495e; font-weight: bold;">{metric}</td>
                            <td style="padding: 8px;">{device[metric]}</td>
                        </tr>
                        """
            elif fallback and fallback in device:
                html += f"""
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px; color: #34495e; font-weight: bold;">Status</td>
                    <td style="padding: 8px;">{device[fallback]}</td>
                </tr>
                """
            else:
                html += f"""
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px; color: #34495e; font-weight: bold;">Status</td>
                    <td style="padding: 8px;">Not available</td>
                </tr>
                """
        
        html += """
            </table>
        </div>
        """
        
        display(HTML(html))
    
    # Driver Information
    if "Driver Version" in gpu_info:
        html = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h3 style="color: #2c3e50; margin-top: 0;">NVIDIA Driver Information</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px; color: #34495e; font-weight: bold;">Driver Version</td>
                    <td style="padding: 8px;">{gpu_info["Driver Version"]}</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px; color: #34495e; font-weight: bold;">NVML Version</td>
                    <td style="padding: 8px;">{gpu_info.get("NVML Version", "Not available")}</td>
                </tr>
            </table>
        </div>
        """
        
        display(HTML(html))

# GPU monitoring class
class GPUMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.keep_monitoring = False
        self.data = {
            'timestamp': [],
            'gpu_util': [],
            'mem_util': [],
            'temperature': [],
            'power_usage': []
        }
        self.nvml_initialized = False
        self.handles = []
    
    def start(self):
        try:
            pynvml.nvmlInit()
            self.nvml_initialized = True
            self.device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(self.device_count):
                self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            
            self.keep_monitoring = True
        except Exception as e:
            print(f"Failed to initialize GPU monitoring: {e}")
            return
    
    def stop(self):
        self.keep_monitoring = False
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self.nvml_initialized = False
            except:
                pass
    
    def monitor(self, duration=60):
        """Monitor GPU for specified duration in seconds"""
        if not self.nvml_initialized:
            self.start()
            if not self.nvml_initialized:
                return pd.DataFrame()
        
        start_time = time.time()
        self.data = {
            'timestamp': [],
            'gpu_util': [],
            'mem_util': [],
            'temperature': [],
            'power_usage': []
        }
        
        try:
            while time.time() - start_time < duration and self.keep_monitoring:
                current_time = time.time() - start_time
                
                # Using first GPU for simplicity, can be extended for multiple GPUs
                handle = self.handles[0]
                
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = 0
                
                self.data['timestamp'].append(current_time)
                self.data['gpu_util'].append(util.gpu)
                self.data['mem_util'].append(util.memory)
                self.data['temperature'].append(temp)
                self.data['power_usage'].append(power)
                
                time.sleep(self.interval)
            
            return pd.DataFrame(self.data)
        
        except Exception as e:
            print(f"Error during GPU monitoring: {e}")
            return pd.DataFrame(self.data)
        
        finally:
            self.stop()
    
    def plot_metrics(self, df=None):
        if df is None:
            if not self.data['timestamp']:
                print("No monitoring data available")
                return
            df = pd.DataFrame(self.data)
        
        if df.empty:
            print("No monitoring data available")
            return
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # GPU Utilization
        axes[0].plot(df['timestamp'], df['gpu_util'], 'b-', linewidth=2)
        axes[0].set_title('GPU Utilization (%)', fontsize=14)
        axes[0].set_ylabel('Utilization (%)')
        axes[0].grid(True)
        axes[0].set_ylim(0, 105)
        
        # Memory Utilization
        axes[1].plot(df['timestamp'], df['mem_util'], 'g-', linewidth=2)
        axes[1].set_title('Memory Utilization (%)', fontsize=14)
        axes[1].set_ylabel('Utilization (%)')
        axes[1].grid(True)
        axes[1].set_ylim(0, 105)
        
        # Temperature
        axes[2].plot(df['timestamp'], df['temperature'], 'r-', linewidth=2)
        axes[2].set_title('GPU Temperature (°C)', fontsize=14)
        axes[2].set_ylabel('Temperature (°C)')
        axes[2].grid(True)
        
        # Power Usage
        if any(df['power_usage'] > 0):
            axes[3].plot(df['timestamp'], df['power_usage'], 'purple', linewidth=2)
            axes[3].set_title('Power Usage (W)', fontsize=14)
            axes[3].set_ylabel('Power (W)')
            axes[3].grid(True)
        else:
            axes[3].text(0.5, 0.5, 'Power usage data not available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[3].transAxes, fontsize=14)
        
        # X-axis label
        axes[3].set_xlabel('Time (seconds)', fontsize=14)
        
        plt.tight_layout()
        return plt

# Modified run_all_tests to include NVML info
def run_all_tests():
    # Display PyTorch GPU info
    display_gpu_info()
    
    # Display NVML detailed GPU info
    display_nvml_gpu_info()
    
    print("\n" + "="*50)
    print("NVIDIA GPU Performance Tests")
    print("="*50 + "\n")
    
    # Create GPU Monitor
    gpu_monitor = GPUMonitor(interval=0.5)
    
    # Matrix multiplication test with monitoring
    print("Starting GPU monitoring for matrix multiplication test...")
    gpu_monitor.start()
    matmul_results = test_matrix_multiplication(sizes=[1000, 2000, 4000], repeat=3)
    matmul_monitor_data = gpu_monitor.monitor(duration=30)  # Monitor for 30 seconds
    
    if not matmul_results.empty:
        print("\n📊 Matrix Multiplication Results:")
        display(matmul_results)
        
        plot = plot_results(matmul_results, 
                           "Matrix Multiplication Performance Comparison", 
                           "Matrix Size", 
                           y_col="Time",
                           hue_col="Device")
        plot.show()
        
        # Plot monitoring data if available
        if not matmul_monitor_data.empty:
            print("\n📈 GPU Metrics During Matrix Multiplication:")
            monitor_plot = gpu_monitor.plot_metrics(matmul_monitor_data)
            monitor_plot.show()
    
    # Rest of the tests with monitoring...
    # (similarly add monitoring to FFT and neural network tests)
    
    # FFT test
    print("Starting GPU monitoring for FFT test...")
    gpu_monitor.start()
    fft_results = test_fft(sizes=[2**20, 2**22, 2**24], repeat=3)
    fft_monitor_data = gpu_monitor.monitor(duration=30)
    
    if not fft_results.empty:
        print("\n📊 FFT Results:")
        display(fft_results)
        
        plot = plot_results(fft_results, 
                           "FFT Performance Comparison", 
                           "FFT Size", 
                           y_col="Time",
                           hue_col="Device")
        plot.show()
        
        # Plot monitoring data if available
        if not fft_monitor_data.empty:
            print("\n📈 GPU Metrics During FFT:")
            monitor_plot = gpu_monitor.plot_metrics(fft_monitor_data)
            monitor_plot.show()
    
    # Neural network test
    print("Starting GPU monitoring for neural network test...")
    gpu_monitor.start()
    nn_results = test_neural_net_ops(batch_sizes=[64, 128, 256], repeat=3)
    nn_monitor_data = gpu_monitor.monitor(duration=60)
    
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
        
        # Plot monitoring data if available
        if not nn_monitor_data.empty:
            print("\n📈 GPU Metrics During Neural Network Operations:")
            monitor_plot = gpu_monitor.plot_metrics(nn_monitor_data)
            monitor_plot.show()
    
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
        
        # Neural network speedup calculation as in the original code
        # ...