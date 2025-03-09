# Add this to your imports
try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False
    print("PyNVML not available. Some GPU metrics will be limited.")

# Function to get detailed GPU information using pynvml
def get_detailed_gpu_info():
    if not pynvml_available:
        return get_gpu_info()  # Fall back to the existing function
        
    gpu_info = {"GPU Available": False, "Details": {}}
    
    try:
        pynvml.nvmlInit()
        gpu_info["GPU Available"] = True
        
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info["Details"]["Device Count"] = device_count
        
        # Get info for each GPU
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Basic info
            gpu_info["Details"][f"GPU {i} Name"] = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Memory info
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info["Details"][f"GPU {i} Memory Total"] = f"{memory.total / 1024**2:.2f} MB"
            gpu_info["Details"][f"GPU {i} Memory Used"] = f"{memory.used / 1024**2:.2f} MB"
            gpu_info["Details"][f"GPU {i} Memory Free"] = f"{memory.free / 1024**2:.2f} MB"
            
            # Utilization info
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_info["Details"][f"GPU {i} Utilization"] = f"{utilization.gpu}%"
            gpu_info["Details"][f"GPU {i} Memory Utilization"] = f"{utilization.memory}%"
            
            # Power info
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # convert to watts
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                gpu_info["Details"][f"GPU {i} Power Usage"] = f"{power_usage:.2f} W"
                gpu_info["Details"][f"GPU {i} Power Limit"] = f"{power_limit:.2f} W"
            except pynvml.NVMLError:
                gpu_info["Details"][f"GPU {i} Power Info"] = "Not available"
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_info["Details"][f"GPU {i} Temperature"] = f"{temp}°C"
            except pynvml.NVMLError:
                gpu_info["Details"][f"GPU {i} Temperature"] = "Not available"
            
            # Clock speeds
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                gpu_info["Details"][f"GPU {i} Graphics Clock"] = f"{graphics_clock} MHz"
                gpu_info["Details"][f"GPU {i} Memory Clock"] = f"{memory_clock} MHz"
            except pynvml.NVMLError:
                gpu_info["Details"][f"GPU {i} Clock Speeds"] = "Not available"
            
        pynvml.nvmlShutdown()
        
    except pynvml.NVMLError as error:
        gpu_info["Error"] = str(error)
        
    return gpu_info

# Function to monitor GPU metrics during performance tests
def monitor_gpu_metrics(callback_function, interval=1.0, duration=None):
    """
    Monitor GPU metrics during the execution of a function
    
    Parameters:
    callback_function: Function to monitor
    interval: Polling interval in seconds
    duration: Maximum monitoring duration in seconds (None for unlimited)
    
    Returns:
    (result, metrics): Result of the callback function and metrics data
    """
    if not pynvml_available:
        print("PyNVML not available. Cannot monitor GPU metrics.")
        return callback_function(), None
    
    import threading
    import time
    
    metrics = []
    stop_monitoring = threading.Event()
    
    def collect_metrics():
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            start_time = time.time()
            elapsed = 0
            
            while not stop_monitoring.is_set() and (duration is None or elapsed < duration):
                timestamp = time.time() - start_time
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = None
                    
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except:
                        power = None
                    
                    metrics.append({
                        'time': timestamp,
                        'gpu_id': i,
                        'gpu_util': utilization.gpu,
                        'mem_util': utilization.memory,
                        'mem_used_mb': memory.used / 1024**2,
                        'temp_c': temp,
                        'power_w': power
                    })
                
                time.sleep(interval)
                elapsed = time.time() - start_time
                
            pynvml.nvmlShutdown()
            
        except pynvml.NVMLError as error:
            print(f"Error monitoring GPU: {error}")
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=collect_metrics)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run the callback function
    try:
        result = callback_function()
    finally:
        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()
    
    return result, pd.DataFrame(metrics) if metrics else None

# Function to plot GPU metrics over time
def plot_gpu_metrics(metrics_df):
    if metrics_df is None or metrics_df.empty:
        print("No GPU metrics available to plot.")
        return
    
    # Check if we have multiple GPUs
    gpu_ids = metrics_df['gpu_id'].unique()
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot GPU Utilization
    for gpu_id in gpu_ids:
        gpu_data = metrics_df[metrics_df['gpu_id'] == gpu_id]
        axs[0].plot(gpu_data['time'], gpu_data['gpu_util'], label=f'GPU {gpu_id}')
    
    axs[0].set_ylabel('GPU Utilization (%)')
    axs[0].set_title('GPU Utilization Over Time')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot Memory Utilization
    for gpu_id in gpu_ids:
        gpu_data = metrics_df[metrics_df['gpu_id'] == gpu_id]
        axs[1].plot(gpu_data['time'], gpu_data['mem_used_mb'], label=f'GPU {gpu_id}')
    
    axs[1].set_ylabel('Memory Used (MB)')
    axs[1].set_title('GPU Memory Usage Over Time')
    axs[1].grid(True)
    axs[1].legend()
    
    # Plot Temperature (if available)
    temp_available = 'temp_c' in metrics_df.columns and not metrics_df['temp_c'].isnull().all()
    power_available = 'power_w' in metrics_df.columns and not metrics_df['power_w'].isnull().all()
    
    if temp_available:
        ax2 = axs[2]
        
        for gpu_id in gpu_ids:
            gpu_data = metrics_df[metrics_df['gpu_id'] == gpu_id]
            ax2.plot(gpu_data['time'], gpu_data['temp_c'], label=f'GPU {gpu_id} Temp')
        
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('GPU Temperature Over Time')
        
        if power_available:
            ax3 = ax2.twinx()
            for gpu_id in gpu_ids:
                gpu_data = metrics_df[metrics_df['gpu_id'] == gpu_id]
                ax3.plot(gpu_data['time'], gpu_data['power_w'], '--', label=f'GPU {gpu_id} Power')
            ax3.set_ylabel('Power (Watts)', color='tab:red')
            ax3.tick_params(axis='y', colors='tab:red')
    
    elif power_available:
        for gpu_id in gpu_ids:
            gpu_data = metrics_df[metrics_df['gpu_id'] == gpu_id]
            axs[2].plot(gpu_data['time'], gpu_data['power_w'], label=f'GPU {gpu_id}')
        
        axs[2].set_ylabel('Power (Watts)')
        axs[2].set_title('GPU Power Usage Over Time')
    
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    return fig