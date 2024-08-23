class AdaptiveKLWeight:
    def __init__(self, init_weight=0.0001, target_kl=0.1, adaptive_rate=0.01):
        self.weight = init_weight
        self.target_kl = target_kl
        self.adaptive_rate = adaptive_rate

    def __call__(self, current_kl):
        self.weight += self.adaptive_rate * (current_kl - self.target_kl)
        return max(0, self.weight)  

# 2. Implement a simple KL annealing schedule
def kl_annealing(epoch, start_weight=0.0001, max_weight=0.01, num_epochs=100):
    return min(max_weight, start_weight + (max_weight - start_weight) * (epoch / num_epochs))