import pynvml
import psutil  # Optional: for getting additional process information

def get_gpu_processes():
    """
    Get information about all processes using GPUs
    
    Returns:
        A list of dictionaries containing process information for each GPU
    """
    # Initialize NVML
    pynvml.nvmlInit()
    
    try:
        # Get device count
        device_count = pynvml.nvmlDeviceGetCount()
        
        # Store all process info
        all_gpu_processes = []
        
        # For each GPU
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            
            gpu_info = {
                "gpu_id": i,
                "name": name,
                "processes": []
            }
            
            # Get all compute processes
            try:
                compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in compute_procs:
                    process_info = {
                        "pid": proc.pid,
                        "memory_used": proc.usedGpuMemory / 1024**2,  # Convert to MB
                        "type": "compute"
                    }
                    
                    # Get process name
                    try:
                        process_info["name"] = pynvml.nvmlSystemGetProcessName(proc.pid)
                    except pynvml.NVMLError:
                        process_info["name"] = "Unknown"
                        
                    # Optional: Get additional process info using psutil
                    try:
                        if psutil:
                            p = psutil.Process(proc.pid)
                            process_info["username"] = p.username()
                            process_info["cpu_percent"] = p.cpu_percent()
                            process_info["create_time"] = p.create_time()
                            process_info["cmdline"] = p.cmdline()
                    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                        pass
                        
                    gpu_info["processes"].append(process_info)
            except pynvml.NVMLError:
                pass
                
            # Get all graphics processes
            try:
                graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                for proc in graphics_procs:
                    # Skip if this PID is already in the list (some processes use both compute and graphics)
                    if any(p["pid"] == proc.pid for p in gpu_info["processes"]):
                        continue
                        
                    process_info = {
                        "pid": proc.pid,
                        "memory_used": proc.usedGpuMemory / 1024**2,  # Convert to MB
                        "type": "graphics"
                    }
                    
                    # Get process name
                    try:
                        process_info["name"] = pynvml.nvmlSystemGetProcessName(proc.pid)
                    except pynvml.NVMLError:
                        process_info["name"] = "Unknown"
                        
                    # Optional: Get additional process info using psutil
                    try:
                        if psutil:
                            p = psutil.Process(proc.pid)
                            process_info["username"] = p.username()
                            process_info["cpu_percent"] = p.cpu_percent()
                            process_info["create_time"] = p.create_time()
                            process_info["cmdline"] = p.cmdline()
                    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                        pass
                        
                    gpu_info["processes"].append(process_info)
            except pynvml.NVMLError:
                pass
                
            all_gpu_processes.append(gpu_info)
            
        return all_gpu_processes
        
    finally:
        # Clean up
        pynvml.nvmlShutdown()

# Example usage
if __name__ == "__main__":
    gpu_processes = get_gpu_processes()
    
    for gpu in gpu_processes:
        print(f"GPU {gpu['gpu_id']}: {gpu['name']}")
        
        if not gpu['processes']:
            print("  No processes using this GPU")
        else:
            print(f"  Found {len(gpu['processes'])} processes:")
            
            for proc in gpu['processes']:
                print(f"    PID: {proc['pid']}")
                print(f"    Name: {proc['name']}")
                print(f"    Memory: {proc['memory_used']:.2f} MB")
                print(f"    Type: {proc['type']}")
                
                # Print additional info if available
                if 'username' in proc:
                    print(f"    User: {proc['username']}")
                if 'cmdline' in proc:
                    print(f"    Command: {' '.join(proc['cmdline'])}")
                    
                print("")