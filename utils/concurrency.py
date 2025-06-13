import asyncio, time

class RateLimiterAsync:
    def __init__(self, rate: int, per: int):
        self.rate, self.per = rate, per
        self.allowance = rate
        self.last = time.time()

    async def acquire(self):
        while True:
            now = time.time()
            elapsed = now - self.last
            self.last = now
            self.allowance += elapsed * (self.rate / self.per)
            if self.allowance > self.rate:
                self.allowance = self.rate
            if self.allowance < 1:
                await asyncio.sleep((1 - self.allowance) * (self.per / self.rate))
            else:
                self.allowance -= 1
                break