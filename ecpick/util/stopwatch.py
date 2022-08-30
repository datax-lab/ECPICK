from datetime import datetime


class StopWatch:
    def __init__(self):
        self.start_datetime = None
        self.end_datetime = None

    def reset(self):
        self.start_datetime = None
        self.end_datetime = None

    def start(self):
        self.reset()
        self.start_datetime = round(datetime.utcnow().timestamp() * 1000)
        return self.start_datetime

    def stop(self):
        return self.get_now()

    def get_now(self):
        self.end_datetime = round(datetime.utcnow().timestamp() * 1000)
        return self.get_elapsed_seconds()

    def get_elapsed_seconds(self):
        return self.end_datetime - self.start_datetime
