import time

class FPS:

    def __init__(self):
        self.prev=time.time()

    def update(self):

        now=time.time()

        fps=1/(now-self.prev)

        self.prev=now

        return int(fps)