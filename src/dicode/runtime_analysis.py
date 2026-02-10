import os
import threading
import time

import matplotlib.pyplot as plt
import pandas as pd


class RuntimeTracker:
	_instance = None
	_lock = threading.Lock()

	def __new__(cls, *args, **kwargs):
		if not cls._instance:
			with cls._lock:
				if not cls._instance:
					cls._instance = super(RuntimeTracker, cls).__new__(cls)
					cls._instance._initialized = False
		return cls._instance

	def __init__(self, output_dir="runtime_analysis"):
		if self._initialized:
			return
		self.output_dir = output_dir
		self.timings = []  # List of dicts: {session: int, component: str, duration: float}
		self.current_timers = {}
		self.lock = threading.Lock()
		self._initialized = True

	def start_timer(self, component_name):
		with self.lock:
			self.current_timers[component_name] = time.time()

	def stop_timer(self, component_name, session_idx):
		with self.lock:
			if component_name in self.current_timers:
				start_time = self.current_timers.pop(component_name)
				duration = time.time() - start_time
				self.timings.append(
					{"session": session_idx, "component": component_name, "duration": duration}
				)
				return duration
			return 0.0

	def log_duration(self, component_name, session_idx, duration):
		with self.lock:
			self.timings.append(
				{"session": session_idx, "component": component_name, "duration": duration}
			)

	def save_data(self):
		with self.lock:
			if not self.timings:
				return
			os.makedirs(self.output_dir, exist_ok=True)
			df = pd.DataFrame(self.timings)
			df.to_csv(os.path.join(self.output_dir, "timings.csv"), index=False)

	def plot_results(self):
		with self.lock:
			if not self.timings:
				return
			df = pd.DataFrame(self.timings)

		if df.empty:
			return

		# Aggregate by session and component (sum in case of multiple calls, though unlikely for these components per session)
		pivot_df = df.pivot_table(
			index="session", columns="component", values="duration", aggfunc="sum"
		)

		plt.figure(figsize=(12, 8))
		pivot_df.plot(kind="bar", stacked=True)
		plt.title("Runtime Breakdown per Session")
		plt.xlabel("Session Index")
		plt.ylabel("Time (seconds)")
		plt.legend(title="Component")
		plt.tight_layout()
		os.makedirs(self.output_dir, exist_ok=True)
		plt.savefig(os.path.join(self.output_dir, "runtime_breakdown.png"))
		plt.close()


# Global instance
tracker = RuntimeTracker()
