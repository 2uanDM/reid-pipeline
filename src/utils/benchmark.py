import threading
import time

import GPUtil
import psutil


class SystemMonitor:
    def __init__(self):
        pass  # No initialization required for psutil or GPUtil

    def get_cpu_usage(self):
        """Get CPU usage percentage."""
        return psutil.cpu_percent()

    def get_ram_usage(self):
        """Get RAM usage as a percentage and total/available RAM."""
        memory = psutil.virtual_memory()
        return {
            "usage_percent": memory.percent,
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
        }

    def get_gpu_usage(self):
        """Get GPU usage and memory for all available GPUs."""
        gpus = GPUtil.getGPUs()
        if not gpus:
            return {"error": "No GPUs found"}

        gpu_usage = []
        for gpu in gpus:
            gpu_usage.append(
                {
                    "name": gpu.name,
                    "gpu_usage_percent": gpu.load * 100,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_total_mb": gpu.memoryTotal,
                }
            )
        return gpu_usage

    def get_usage_dict(self):
        """Return CPU, RAM, and GPU usage as a dictionary."""
        usage = {
            "cpu_usage_percent": self.get_cpu_usage(),
            "ram": self.get_ram_usage(),
            "gpu": self.get_gpu_usage(),
        }
        return usage

    def print_usage(self):
        """Print CPU, RAM, and GPU usage."""
        usage = self.get_usage_dict()

        print("System Resource Usage:")
        print(f"CPU Usage: {usage['cpu_usage_percent']}%")
        print(f"RAM Usage: {usage['ram']['usage_percent']}%")
        print(f"Total RAM: {usage['ram']['total_gb']:.2f} GB")
        print(f"Available RAM: {usage['ram']['available_gb']:.2f} GB")

        gpu_usage = usage["gpu"]
        if isinstance(gpu_usage, dict) and "error" in gpu_usage:
            print(f"GPU Monitoring: {gpu_usage['error']}")
        else:
            for i, gpu in enumerate(gpu_usage):
                print(f"GPU {i + 1} ({gpu['name']}):")
                print(f"  GPU Usage: {gpu['gpu_usage_percent']:.2f}%")
                print(
                    f"  GPU Memory: {gpu['memory_used_mb']:.2f} MB / {gpu['memory_total_mb']:.2f} MB"
                )


class BenchmarkMultiThread:
    def __init__(self):
        self.monitor = SystemMonitor()
        self.start_time = time.time()
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.lock = (
            threading.Lock()
        )  # To safely access shared resources in multithreading
        self.running = False

    def log_metrics(self):
        """Log the current system metrics."""
        while self.running:
            usage = self.monitor.get_usage_dict()
            with self.lock:  # Ensure thread-safe access to shared lists
                self.cpu_usage.append(usage["cpu_usage_percent"])
                self.ram_usage.append(usage["ram"]["usage_percent"])

                # Calculate average GPU usage percentage across all GPUs
                gpu_data = usage["gpu"]
                if isinstance(gpu_data, dict) and "error" in gpu_data:
                    gpu_avg_usage = 0  # Set to 0 if no GPUs found
                else:
                    gpu_avg_usage = sum(
                        gpu["gpu_usage_percent"] for gpu in gpu_data
                    ) / len(gpu_data)
                self.gpu_usage.append(gpu_avg_usage)

            time.sleep(1)  # Log metrics every 1 second

    def calculate_averages(self):
        """Calculate averages of logged metrics."""
        with self.lock:  # Ensure thread-safe access to shared lists
            avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
            avg_ram = sum(self.ram_usage) / len(self.ram_usage) if self.ram_usage else 0
            avg_gpu = sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0
        elapsed_time = time.time() - self.start_time
        return {
            "time_elapsed": elapsed_time,
            "cpu_avg_percent": avg_cpu,
            "ram_avg_percent": avg_ram,
            "gpu_avg_percent": avg_gpu,
        }

    def start_logging(self):
        """Start logging metrics in a separate thread."""
        self.running = True
        self.logging_thread = threading.Thread(target=self.log_metrics, daemon=True)
        self.logging_thread.start()

    def stop_logging(self):
        """Stop logging metrics."""
        self.running = False
        self.logging_thread.join()  # Wait for the logging thread to finish

    def example(self, duration):
        """Run the benchmark for a specified duration."""
        self.start_logging()
        print("Benchmark started...")

        try:
            time.sleep(
                duration
            )  # Allow the benchmark to run for the specified duration
        finally:
            self.stop_logging()  # Ensure logging stops even if interrupted

        averages = self.calculate_averages()
        print("Benchmark Complete.")
        print(
            f"Time: {averages['time_elapsed']:.2f}s, "
            f"CPU Avg: {averages['cpu_avg_percent']:.2f}%, "
            f"RAM Avg: {averages['ram_avg_percent']:.2f}%, "
            f"GPU Avg: {averages['gpu_avg_percent']:.2f}%"
        )
