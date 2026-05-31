import numpy as np
import os

class RunnerM():

    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        save_name = kwargs.get("save_name", "best_model.pickle")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        best_score = 0

        for epoch in range(num_epochs):

            X, y = train_set
            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            num_batches = (X.shape[0] + self.batch_size - 1) // self.batch_size
            
            epoch_train_loss = 0.0
            epoch_train_score = 0.0

            for iteration in range(num_batches):

                train_X = X[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration + 1) * self.batch_size]

                logits = self.model(train_X)
                
                trn_loss = self.loss_fn(logits, train_y)
                trn_score = self.metric(logits, train_y)

                epoch_train_loss += trn_loss * train_X.shape[0]
                epoch_train_score += trn_score * train_X.shape[0]

                self.loss_fn.backward()
                self.optimizer.step()
                self.model.clear_grad()

                if (iteration) % log_iters == 0:
                    print(f"Epoch: {epoch}, Iteration: {iteration} | [Train Batch] loss: {trn_loss:.4f}, score: {trn_score:.4f}")
            
            avg_train_loss = epoch_train_loss / X.shape[0]
            avg_train_score = epoch_train_score / X.shape[0]
            
            self.train_loss.append(avg_train_loss)
            self.train_scores.append(avg_train_score)

            dev_score, dev_loss = self.evaluate(dev_set)

            self.dev_scores.append(dev_score)
            self.dev_loss.append(dev_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            print(f"=== Epoch {epoch} Summary ===")
            print(f"[Train] avg_loss: {avg_train_loss:.5f}, avg_score: {avg_train_score:.5f}")
            print(f"[Dev]   avg_loss: {dev_loss:.5f}, avg_score: {dev_score:.5f}")

            if dev_score > best_score:
                save_path = os.path.join(save_dir, save_name)
                self.save_model(save_path)
                print(f"⭐ Best accuracy performance updated: {best_score:.5f} --> {dev_score:.5f}\n")
                best_score = dev_score
            else:
                print()

        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        total_loss = 0
        total_score = 0

        num_batches = (X.shape[0] + self.batch_size - 1) // self.batch_size

        for iteration in range(num_batches):
            batch_X = X[iteration * self.batch_size : (iteration + 1) * self.batch_size]
            batch_y = y[iteration * self.batch_size : (iteration + 1) * self.batch_size]

            logits = self.model(batch_X)
            loss = self.loss_fn(logits, batch_y)
            score = self.metric(logits, batch_y)

            total_loss += loss * batch_X.shape[0]
            total_score += score * batch_X.shape[0]

        total_loss /= X.shape[0]
        total_score /= X.shape[0]

        return total_score, total_loss

    def save_model(self, save_path):
        self.model.save_model(save_path)