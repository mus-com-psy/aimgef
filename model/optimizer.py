class CustomSchedule:
    def __init__(self, model_size, factor, warm_up, optimizer):
        super(CustomSchedule, self).__init__()
        self.optimizer = optimizer
        self.n_step = 0
        self.warm_up = warm_up
        self.factor = factor
        self.model_size = model_size
        self.n_rate = 0

    def step(self):
        self.n_step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.n_rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self.n_step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warm_up ** (-1.5)))

    def state_dict(self):
        return {'opt_state_dict': self.optimizer.state_dict(), 'step': self.n_step, 'rate': self.n_rate}

    def load_state_dict(self, state_dict):
        self.n_step = state_dict['step']
        self.n_rate = state_dict['rate']
        self.optimizer.load_state_dict(state_dict['opt_state_dict'])
