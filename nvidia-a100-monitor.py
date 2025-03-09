import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
import subprocess
import re
import io
import seaborn as sns

class A100Monitor:
    def __init__(self, update_interval=1.0, max_points=100):
        """
        Initialize the A100 GPU monitor
        
        Parameters:
        -----------
        update_interval : float
            Time between updates in seconds
        max_points : int
            Maximum number of data points to store in history
        """
        self.update_interval = update_interval
        self.max_points = max_points
        self.fig = None
        self.ani = None
        
        # Initialize data storage
        self.timestamps = []
        self.metrics = {
            'gpu_util': [],
            'mem_util': [],
            'temperature': [],
            'power_usage': [],
            'mem_used': [],
            'mem_total': []
        }
        
        # Check if we have access to nvidia-smi
        try:
            subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
            self.gpu_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.gpu_available = False
            print("Warning: nvidia-smi not available or no NVIDIA GPU detected")

    def get_gpu_metrics(self):
        """Get current GPU metrics using nvidia-smi"""
        if not self.gpu_available:
            # Return dummy data if GPU not available
            return {
                'gpu_util': 0,
                'mem_util': 0,
                'temperature': 0,
                'power_usage': 0,
                'mem_used': 0,
                'mem_total': 1
            }
            
        # Query nvidia-smi for GPU metrics
        output = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ]).decode('utf-8')
        
        # Parse the output
        values = output.strip().split(',')
        if len(values) >= 6:
            return {
                'gpu_util': float(values[0]),
                'mem_util': float(values[1]),
                'temperature': float(values[2]),
                'power_usage': float(values[3]),
                'mem_used': float(values[4]),
                'mem_total': float(values[5])
            }
        else:
            # Return zeros if we couldn't parse the output
            return {key: 0 for key in self.metrics.keys()}
    
    def update_data(self):
        """Update the data with current metrics"""
        current_metrics = self.get_gpu_metrics()
        
        # Add current timestamp
        self.timestamps.append(time.time())
        
        # Update all metrics
        for key, value in current_metrics.items():
            self.metrics[key].append(value)
        
        # Keep only the most recent max_points
        if len(self.timestamps) > self.max_points:
            self.timestamps = self.timestamps[-self.max_points:]
            for key in self.metrics:
                self.metrics[key] = self.metrics[key][-self.max_points:]
    
    def create_dataframe(self):
        """Create a pandas DataFrame from current metrics"""
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'GPU Utilization (%)': self.metrics['gpu_util'],
            'Memory Utilization (%)': self.metrics['mem_util'],
            'Temperature (°C)': self.metrics['temperature'],
            'Power Usage (W)': self.metrics['power_usage'],
            'Memory Used (MB)': self.metrics['mem_used'],
            'Memory Total (MB)': self.metrics['mem_total']
        })
        # Add derived column
        df['Memory Used (%)'] = (df['Memory Used (MB)'] / df['Memory Total (MB)']) * 100
        # Add relative time
        df['Relative Time (s)'] = df['timestamp'] - df['timestamp'].iloc[0]
        return df
    
    def display_metrics(self):
        """Display current metrics in a formatted table"""
        if len(self.timestamps) == 0:
            return pd.DataFrame()
        
        df = self.create_dataframe().iloc[-1:].copy()
        # Format the display
        display_df = pd.DataFrame({
            'Metric': ['GPU Utilization', 'Memory Utilization', 'Temperature', 
                      'Power Usage', 'Memory Used', 'Memory Total'],
            'Value': [
                f"{df['GPU Utilization (%)'].iloc[0]:.1f}%",
                f"{df['Memory Utilization (%)'].iloc[0]:.1f}%",
                f"{df['Temperature (°C)'].iloc[0]:.1f}°C",
                f"{df['Power Usage (W)'].iloc[0]:.1f}W",
                f"{df['Memory Used (MB)'].iloc[0]:.1f}MB ({df['Memory Used (%)'].iloc[0]:.1f}%)",
                f"{df['Memory Total (MB)'].iloc[0]:.1f}MB"
            ]
        })
        return display_df
    
    def plot_metrics(self):
        """Create plots for the metrics"""
        if len(self.timestamps) < 2:
            return
        
        df = self.create_dataframe()
        
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('NVIDIA A100 GPU Performance Metrics', fontsize=16)
        
        # Plot GPU and Memory Utilization
        axs[0, 0].plot(df['Relative Time (s)'], df['GPU Utilization (%)'], 'b-', label='GPU')
        axs[0, 0].plot(df['Relative Time (s)'], df['Memory Used (%)'], 'g-', label='Memory')
        axs[0, 0].set_title('Utilization (%)')
        axs[0, 0].set_ylabel('Percentage (%)')
        axs[0, 0].set_ylim(0, 105)
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Plot Temperature
        axs[0, 1].plot(df['Relative Time (s)'], df['Temperature (°C)'], 'r-')
        axs[0, 1].set_title('GPU Temperature')
        axs[0, 1].set_ylabel('Temperature (°C)')
        axs[0, 1].grid(True)
        
        # Plot Power Usage
        axs[1, 0].plot(df['Relative Time (s)'], df['Power Usage (W)'], 'c-')
        axs[1, 0].set_title('Power Usage')
        axs[1, 0].set_ylabel('Power (W)')
        axs[1, 0].set_xlabel('Time (seconds)')
        axs[1, 0].grid(True)
        
        # Plot Memory Usage
        axs[1, 1].plot(df['Relative Time (s)'], df['Memory Used (MB)'], 'm-')
        axs[1, 1].set_title('Memory Usage')
        axs[1, 1].set_ylabel('Memory (MB)')
        axs[1, 1].set_xlabel('Time (seconds)')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def update_plot(self, frame):
        """Update function for animation"""
        self.update_data()
        
        # Display current metrics
        display_df = self.display_metrics()
        clear_output(wait=True)
        display(display_df)
        
        # Update plots
        fig = self.plot_metrics()
        plt.close(fig)  # Close to prevent memory leaks
        return fig
    
    def start_monitoring(self):
        """Start the monitoring with live updates"""
        if not self.gpu_available:
            print("No NVIDIA GPU detected. Running in simulation mode.")
            
        # Create initial plot
        self.update_data()
        self.fig = self.plot_metrics()
        plt.close(self.fig)
        
        # Create animation
        self.ani = FuncAnimation(
            plt.figure(figsize=(14, 10)),
            self.update_plot,
            interval=self.update_interval * 1000,  # Convert to milliseconds
            cache_frame_data=False
        )
        
        # Display initial metrics
        display(self.display_metrics())
        
        return self.ani
    
    def get_gpu_info(self):
        """Get and display information about the GPU"""
        if not self.gpu_available:
            return "No NVIDIA GPU detected."
            
        try:
            # Get GPU name and driver version
            name_output = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).decode('utf-8').strip()
            driver_output = subprocess.check_output(['nvidia-smi', '--query', '--display=DRIVER', '--format=csv,noheader']).decode('utf-8')
            driver_version = re.search(r'Driver Version\s+:\s+([\d\.]+)', driver_output)
            driver_version = driver_version.group(1) if driver_version else "Unknown"
            
            # Get CUDA version
            cuda_output = subprocess.check_output(['nvidia-smi', '--query', '--display=COMPUTE', '--format=csv,noheader']).decode('utf-8')
            cuda_version = re.search(r'CUDA Version\s+:\s+([\d\.]+)', cuda_output)
            cuda_version = cuda_version.group(1) if cuda_version else "Unknown"
            
            return f"""
            GPU: {name_output}
            Driver Version: {driver_version}
            CUDA Version: {cuda_version}
            """
        except Exception as e:
            return f"Error getting GPU info: {str(e)}"

# Usage example in Jupyter Notebook
monitor = A100Monitor(update_interval=2.0, max_points=100)
print(monitor.get_gpu_info())
animation = monitor.start_monitoring()

# To stop monitoring when done
# animation.event_source.stop()
