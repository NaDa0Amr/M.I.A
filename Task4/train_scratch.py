"""Train the scratch MLP on scikit-learn digits (8x8) dataset."""
import random
import math
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from scratchnn import MLP, Value


def one_hot(k, K=10):
    v = [0.0]*K
    v[k] = 1.0
    return v


def mse_loss(preds, targets):
    # preds: list of Value outputs
    loss = Value(0.0)
    for p, t in zip(preds, targets):
        diff = p - t
        loss = loss + diff * diff
    return loss


def predict(model, X):
    ypred = []
    for x in X:
        out = model(x, training=False)
        vals = [o.data for o in out]
        ypred.append(int(np.argmax(vals)))
    return ypred


def main(epochs=20, lr=0.01):
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLP(64, [32, 10], activation='tanh', dropout_p=0.1)

    for epoch in range(epochs):
        # simple SGD
        total_loss = 0.0
        for xi, yi in zip(Xtrain, ytrain):
            preds = model(xi, training=True)
            tgt = [Value(v) for v in one_hot(yi, 10)]
            loss = mse_loss(preds, tgt)
            # zero grads
            for p in model.parameters():
                p.grad = 0.0
            loss.backward()
            total_loss += loss.data
            # update
            for p in model.parameters():
                p.data += -lr * p.grad

        # eval
        ypred = predict(model, Xtest)
        acc = accuracy_score(ytest, ypred)
        print(f"Epoch {epoch+1}/{epochs} loss={total_loss/len(Xtrain):.4f} acc={acc:.4f}")

    cm = confusion_matrix(ytest, ypred)
    print("Confusion matrix:\n", cm)


if __name__ == '__main__':
    main()
