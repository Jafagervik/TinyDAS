class LossScaler:
    def __init__(self, initial_scale=2**15, scale_factor=2, scale_window=2000):
        self.scale = initial_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.step = 0
        self.inf_or_nan_count = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_gradients(self, optimizer):
        for param in optimizer.params:
            if param.grad is not None:
                param.grad /= self.scale

    def update_scale(self, has_inf_or_nan):
        self.step += 1
        if has_inf_or_nan:
            self.inf_or_nan_count += 1
            self.scale /= self.scale_factor
            self.step = 0
            self.inf_or_nan_count = 0
        elif self.step % self.scale_window == 0:
            if self.inf_or_nan_count == 0:
                self.scale *= self.scale_factor
            self.step = 0
            self.inf_or_nan_count = 0