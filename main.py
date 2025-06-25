import ray
import time

@ray.remote
def worker_func(pid):
    time.sleep(5)
    return f"pid {pid} finished"

ray.init(include_dashboard=False, ignore_reinit_error=True)
start = time.time()
results = [worker_func.remote(i) for i in range(3)]
print(ray.get(results))
print("Elapsed:", time.time() - start)