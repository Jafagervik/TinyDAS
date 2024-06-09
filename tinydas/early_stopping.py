class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last time the validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


if __name__ == "__main__":
    early_stopping = EarlyStopping(patience=3, min_delta=0.0)
    losses = [10, 9, 8, 7, 7, 7, 7, 7, 7, 7]
    for i, loss in enumerate(losses):
        print(f"Epoch {i+1}: loss={loss}")
        early_stopping(loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
