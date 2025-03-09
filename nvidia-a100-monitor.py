import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re
from IPython.display import display, HTML, clear_output
import seaborn as sns

class MultiGPUMonitor:
    def __init__(self, update_interval=1.0, max_points=100):
        """
        Initialize the Multi-GPU monitor
        
        Parameters:
        -----------
        update_interval : float
            Time between updates in seconds
        max_points : int
            Maximum number of data points to store in history
        """
        self.update_interval = update_interval
        self.max_points = max_points
        self.running = False
        self.gpu_count = 0
        self.gpu_names = []
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # Check if we have access to nvidia-smi and count GPUs
        try:
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits']).decode('utf-8').strip()
            self.gpu_count = int(output)
            
            # Get GPU names
            names_output = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).decode('utf-8').strip()
            self.gpu_names = names_output.split('\n')
            
            self.gpu_available = True
            print(f"Detected {self.gpu_count} NVIDIA GPUs")
            for i, name in enumerate(self.gpu_names):
                print(f"GPU {i}: {name}")
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            self.gpu_available = False
            self.gpu_count = 2  # Default to 2 simulated GPUs
            self.gpu_names = ["Simulated GPU 0", "Simulated GPU 1"]
            print("Warning: nvidia-smi not available or no NVIDIA GPUs detected")
            print(f"Running in simulation mode with {self.gpu_count} synthetic GPUs")
        
        # Initialize data storage - one set per GPU
        self.initialize_data_storage()
    
    def initialize_data_storage(self):
        """Initialize data structures for all GPUs"""
        self.timestamps = []
        # Dictionary of metrics, keys are GPU indices
        self.metrics = {}
        
        for gpu_idx in range(self.gpu_count):
            self.metrics[gpu_idx] = {
                'gpu_util': [],
                'mem_util': [],
                'temperature': [],
                'power_usage': [],
                'mem_used': [],
                'mem_total': []
            }

    def get_gpu_metrics(self):
        """Get current metrics for all GPUs using nvidia-smi"""
        results = {}
        
        if not self.gpu_available:
            # Return simulated data for each GPU
            for gpu_idx in range(self.gpu_count):
                # Add some variation between GPUs
                variation = (gpu_idx + 1) * 10
                results[gpu_idx] = {
                    'gpu_util': np.random.randint(30, 90 - variation),
                    'mem_util': np.random.randint(20, 80 - variation),
                    'temperature': np.random.randint(50, 85 - variation//2),
                    'power_usage': np.random.randint(100, 300 - variation),
                    'mem_used': np.random.randint(5000, 20000),
                    'mem_total': 40000
                }
            return results
            
        # Query nvidia-smi for all GPU metrics
        try:
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ]).decode('utf-8')
            
            # Parse each line (one per GPU)
            for line in output.strip().split('\n'):
                values = line.strip().split(',')
                if len(values) >= 7:
                    gpu_idx = int(values[0])
                    results[gpu_idx] = {
                        'gpu_util': float(values[1]),
                        'mem_util': float(values[2]),
                        'temperature': float(values[3]),
                        'power_usage': float(values[4]),
                        'mem_used': float(values[5]),
                        'mem_total': float(values[6])
                    }
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
            # Return zeros if we couldn't parse the output
            for gpu_idx in range(self.gpu_count):
                results[gpu_idx] = {key: 0 for key in self.metrics[0].keys()}
            
        return results
    
    def update_data(self):
        """Update the data with current metrics for all GPUs"""
        # Get metrics for all GPUs
        all_metrics = self.get_gpu_metrics()
        
        # Add current timestamp
        self.timestamps.append(time.time())
        
        # Update all metrics for each GPU
        for gpu_idx, gpu_metrics in all_metrics.items():
            for key, value in gpu_metrics.items():
                self.metrics[gpu_idx][key].append(value)
        
        # Keep only the most recent max_points
        if len(self.timestamps) > self.max_points:
            self.timestamps = self.timestamps[-self.max_points:]
            for gpu_idx in self.metrics:
                for key in self.metrics[gpu_idx]:
                    self.metrics[gpu_idx][key] = self.metrics[gpu_idx][key][-self.max_points:]
    
    def create_dataframe(self, gpu_idx):
        """Create a pandas DataFrame from current metrics for a specific GPU"""
        if len(self.timestamps) == 0:
            # Return empty DataFrame if no data
            return pd.DataFrame()
            
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'GPU Utilization (%)': self.metrics[gpu_idx]['gpu_util'],
            'Memory Utilization (%)': self.metrics[gpu_idx]['mem_util'],
            'Temperature (°C)': self.metrics[gpu_idx]['temperature'],
            'Power Usage (W)': self.metrics[gpu_idx]['power_usage'],
            'Memory Used (MB)': self.metrics[gpu_idx]['mem_used'],
            'Memory Total (MB)': self.metrics[gpu_idx]['mem_total']
        })
        
        # Add derived column
        df['Memory Used (%)'] = (df['Memory Used (MB)'] / df['Memory Total (MB)']) * 100
        # Add relative time
        df['Relative Time (s)'] = df['timestamp'] - df['timestamp'].iloc[0]
        return df
    
    def display_metrics_table(self):
        """Display current metrics for all GPUs in a formatted table"""
        if len(self.timestamps) == 0:
            return pd.DataFrame()
        
        # Create a list to hold all rows
        rows = []
        
        # For each GPU, add its metrics as rows
        for gpu_idx in range(self.gpu_count):
            if gpu_idx not in self.metrics:
                continue
                
            df = self.create_dataframe(gpu_idx)
            if df.empty:
                continue
                
            latest = df.iloc[-1]
            
            # Add rows for this GPU
            rows.append({
                'GPU': f"{gpu_idx}: {self.gpu_names[gpu_idx]}",
                'Metric': 'GPU Utilization',
                'Value': f"{latest['GPU Utilization (%)']:.1f}%"
            })
            rows.append({
                'GPU': f"{gpu_idx}: {self.gpu_names[gpu_idx]}",
                'Metric': 'Memory Utilization',
                'Value': f"{latest['Memory Used (%)']:.1f}%"
            })
            rows.append({
                'GPU': f"{gpu_idx}: {self.gpu_names[gpu_idx]}",
                'Metric': 'Temperature',
                'Value': f"{latest['Temperature (°C)']:.1f}°C"
            })
            rows.append({
                'GPU': f"{gpu_idx}: {self.gpu_names[gpu_idx]}",
                'Metric': 'Power Usage',
                'Value': f"{latest['Power Usage (W)']:.1f}W"
            })
            rows.append({
                'GPU': f"{gpu_idx}: {self.gpu_names[gpu_idx]}",
                'Metric': 'Memory Used',
                'Value': f"{latest['Memory Used (MB)']:.1f}MB / {latest['Memory Total (MB)']:.1f}MB"
            })
            
            # Add a separator row except for the last GPU
            if gpu_idx < self.gpu_count - 1:
                rows.append({
                    'GPU': '---',
                    'Metric': '---',
                    'Value': '---'
                })
        
        # Convert to DataFrame for display
        display_df = pd.DataFrame(rows)
        return display_df
    
    def plot_comparison(self, metric):
        """Create a plot comparing a specific metric across all GPUs"""
        if len(self.timestamps) < 2:
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f'GPU Comparison: {metric}', fontsize=16)
        
        # Plot the metric for each GPU
        for gpu_idx in range(self.gpu_count):
            if gpu_idx not in self.metrics:
                continue
                
            df = self.create_dataframe(gpu_idx)
            if df.empty:
                continue
                
            if metric == 'GPU Utilization':
                ax.plot(df['Relative Time (s)'], df['GPU Utilization (%)'], 
                        label=f'GPU {gpu_idx}: {self.gpu_names[gpu_idx]}')
                ax.set_ylabel('Utilization (%)')
                ax.set_ylim(0, 105)
            elif metric == 'Memory Utilization':
                ax.plot(df['Relative Time (s)'], df['Memory Used (%)'], 
                        label=f'GPU {gpu_idx}: {self.gpu_names[gpu_idx]}')
                ax.set_ylabel('Memory Utilization (%)')
                ax.set_ylim(0, 105)
            elif metric == 'Temperature':
                ax.plot(df['Relative Time (s)'], df['Temperature (°C)'], 
                        label=f'GPU {gpu_idx}: {self.gpu_names[gpu_idx]}')
                ax.set_ylabel('Temperature (°C)')
            elif metric == 'Power':
                ax.plot(df['Relative Time (s)'], df['Power Usage (W)'], 
                        label=f'GPU {gpu_idx}: {self.gpu_names[gpu_idx]}')
                ax.set_ylabel('Power (W)')
        
        ax.set_xlabel('Time (seconds)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_gpu_details(self, gpu_idx):
        """Create detailed plots for a specific GPU"""
        if len(self.timestamps) < 2 or gpu_idx not in self.metrics:
            return None
        
        df = self.create_dataframe(gpu_idx)
        if df.empty:
            return None
        
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'GPU {gpu_idx}: {self.gpu_names[gpu_idx]} - Performance Metrics', fontsize=16)
        
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
    
    def update_display(self):
        """Update the display with current metrics and plots for all GPUs"""
        self.update_data()
        
        # Clear previous output
        clear_output(wait=True)
        
        # Display current metrics for all GPUs
        display_df = self.display_metrics_table()
        display(display_df)
        
        # Plot comparison charts
        metrics_to_compare = ['GPU Utilization', 'Memory Utilization', 'Temperature', 'Power']
        for metric in metrics_to_compare:
            fig = self.plot_comparison(metric)
            if fig:
                display(fig)
                plt.close(fig)  # Close to prevent memory leaks
        
        # Plot detailed charts for each GPU
        for gpu_idx in range(self.gpu_count):
            fig = self.plot_gpu_details(gpu_idx)
            if fig:
                display(fig)
                plt.close(fig)  # Close to prevent memory leaks
    
    def start_monitoring(self):
        """Start the monitoring with manual updates"""
        if not self.gpu_available:
            print("No NVIDIA GPUs detected. Running in simulation mode.")
        
        print(f"Starting monitoring for {self.gpu_count} GPUs...")
        self.running = True
        
        # Initial update
        self.update_display()
        
        # Display instructions
        print("\nMonitoring started. Run the next cell to update the display.")
        
        return self
    
    def update(self):
        """Manual update function - call this to refresh the display"""
        if self.running:
            self.update_display()
        else:
            print("Monitoring is not running. Call start_monitoring() first.")
    
    def stop_monitoring(self):
        """Stop the monitoring"""
        self.running = False
        print("GPU monitoring stopped.")
    
    def get_gpu_info(self):
        """Get and display information about all GPUs"""
        if not self.gpu_available:
            return "No NVIDIA GPUs detected. Running in simulation mode."
            
        try:
            # Get detailed GPU and system info
            info_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            return info_output
        except Exception as e:
            return f"Error getting GPU info: {str(e)}"
