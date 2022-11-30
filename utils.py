import tensorflow as tf
import matplotlib.pyplot as plt


def plot_accuracy(history):
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.legend()
    plt.show()


def plot_loss(history):
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.legend()
    plt.show()


def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label
