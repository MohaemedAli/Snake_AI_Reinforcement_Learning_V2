class EpsilonScheduler:
    """Linear epsilon decay from `start` to `end` over `decay_steps` steps."""

    def __init__(
        self,
        start=1.0,
        end=0.02,
        decay_steps=40000
    ):

        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def get_epsilon(self, step):

        progress = min(step / self.decay_steps, 1.0)

        return self.start + progress * (self.end - self.start)
