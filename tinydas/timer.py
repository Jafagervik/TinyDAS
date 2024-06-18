from time import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.interval = None

    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time()
        self.interval = self.end_time - self.start_time
        # Return False to propagate any exceptions
        return False
    