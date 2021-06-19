import collections
import os
import time
import os

import matplotlib.pyplot as plt
import numpy as np

import nn

use_graphics = True

def maybe_sleep_and_close(seconds):
    if use_graphics and plt.get_fignums():
        time.sleep(seconds)
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            plt.close(fig)
            try:
                # This raises a TclError on some Windows machines
                fig.canvas.start_event_loop(1e-3)
            except:
                pass

def get_data_path(filename):
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise Exception("Could not find data file: {}".format(filename))
    return path

class Dataset(object):
    def __init__(self, x, y):
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert np.issubdtype(x.dtype, np.floating)
        assert np.issubdtype(y.dtype, np.floating)
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def iterate_once(self, batch_size):
        assert isinstance(batch_size, int) and batch_size > 0, (
            "Batch size should be a positive integer, got {!r}".format(
                batch_size))
        assert self.x.shape[0] % batch_size == 0, (
            "Dataset size {:d} is not divisible by batch size {:d}".format(
                self.x.shape[0], batch_size))
        index = 0
        while index < self.x.shape[0]:
            x = self.x[index:index + batch_size]
            y = self.y[index:index + batch_size]
            yield nn.Constant(x), nn.Constant(y)
            index += batch_size

    def iterate_forever(self, batch_size):
        while True:
            yield from self.iterate_once(batch_size)

    def get_validation_accuracy(self):
        raise NotImplementedError(
            "No validation data is available for this dataset. "
            "In this assignment, only the Digit Classification and Language "
            "Identification datasets have validation data.")

class DigitClassificationDataset(Dataset):
    def __init__(self, model):
        mnist_path = get_data_path("mnist.npz")

        with np.load(mnist_path) as data:
            train_images = data["train_images"]
            train_labels = data["train_labels"]
            test_images = data["test_images"]
            test_labels = data["test_labels"]
            assert len(train_images) == len(train_labels) == 60000
            assert len(test_images) == len(test_labels) == 10000
            self.dev_images = test_images[0::2]
            self.dev_labels = test_labels[0::2]
            self.test_images = test_images[1::2]
            self.test_labels = test_labels[1::2]

        train_labels_one_hot = np.zeros((len(train_images), 10))
        train_labels_one_hot[range(len(train_images)), train_labels] = 1

        super().__init__(train_images, train_labels_one_hot)

        self.model = model
        self.epoch = 0

        if use_graphics:
            width = 20  # Width of each row expressed as a multiple of image width
            samples = 100  # Number of images to display per label
            fig = plt.figure()
            ax = {}
            images = collections.defaultdict(list)
            texts = collections.defaultdict(list)
            for i in reversed(range(10)):
                ax[i] = plt.subplot2grid((30, 1), (3 * i, 0), 2, 1,
                                         sharex=ax.get(9))
                plt.setp(ax[i].get_xticklabels(), visible=i == 9)
                ax[i].set_yticks([])
                ax[i].text(-0.03, 0.5, i, transform=ax[i].transAxes,
                           va="center")
                ax[i].set_xlim(0, 28 * width)
                ax[i].set_ylim(0, 28)
                for j in range(samples):
                    images[i].append(ax[i].imshow(
                        np.zeros((28, 28)), vmin=0, vmax=1, cmap="Greens",
                        alpha=0.3))
                    texts[i].append(ax[i].text(
                        0, 0, "", ha="center", va="top", fontsize="smaller"))
            ax[9].set_xticks(np.linspace(0, 28 * width, 11))
            ax[9].set_xticklabels(
                ["{:.1f}".format(num) for num in np.linspace(0, 1, 11)])
            ax[9].tick_params(axis="x", pad=16)
            ax[9].set_xlabel("Probability of Correct Label")
            status = ax[0].text(
                0.5, 1.5, "", transform=ax[0].transAxes, ha="center",
                va="bottom")
            plt.show(block=False)

            self.width = width
            self.samples = samples
            self.fig = fig
            self.images = images
            self.texts = texts
            self.status = status
            self.last_update = time.time()

    def iterate_once(self, batch_size):
        self.epoch += 1

        for i, (x, y) in enumerate(super().iterate_once(batch_size)):
            yield x, y

            if use_graphics and time.time() - self.last_update > 1:
                dev_logits = self.model.run(nn.Constant(self.dev_images)).data
                dev_predicted = np.argmax(dev_logits, axis=1)
                dev_probs = np.exp(nn.SoftmaxLoss.log_softmax(dev_logits))
                dev_accuracy = np.mean(dev_predicted == self.dev_labels)

                self.status.set_text(
                    "epoch: {:d}, batch: {:d}/{:d}, validation accuracy: "
                    "{:.2%}".format(
                        self.epoch, i, len(self.x) // batch_size, dev_accuracy))
                for i in range(10):
                    predicted = dev_predicted[self.dev_labels == i]
                    probs = dev_probs[self.dev_labels == i][:, i]
                    linspace = np.linspace(
                        0, len(probs) - 1, self.samples).astype(int)
                    indices = probs.argsort()[linspace]
                    for j, (prob, image) in enumerate(zip(
                            probs[indices],
                            self.dev_images[self.dev_labels == i][indices])):
                        self.images[i][j].set_data(image.reshape((28, 28)))
                        left = prob * (self.width - 1) * 28
                        if predicted[indices[j]] == i:
                            self.images[i][j].set_cmap("Greens")
                            self.texts[i][j].set_text("")
                        else:
                            self.images[i][j].set_cmap("Reds")
                            self.texts[i][j].set_text(predicted[indices[j]])
                            self.texts[i][j].set_x(left + 14)
                        self.images[i][j].set_extent([left, left + 28, 0, 28])
                self.fig.canvas.draw_idle()
                self.fig.canvas.start_event_loop(1e-3)
                self.last_update = time.time()

    def get_validation_accuracy(self):
        dev_logits = self.model.run(nn.Constant(self.dev_images)).data
        dev_predicted = np.argmax(dev_logits, axis=1)
        dev_accuracy = np.mean(dev_predicted == self.dev_labels)
        return dev_accuracy

def main():
    import models
    model = models.DigitClassificationModel()
    dataset = DigitClassificationDataset(model)
    model.train(dataset)

if __name__ == "__main__":
    main()
