from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    BatchNormalization,
    Dropout,
)
from utils import plot_accuracy, plot_loss, process


train_ds = image_dataset_from_directory(
    directory="sports-classification/train",
    labels="inferred",
    label_mode="int",
    batch_size=64,
    image_size=(256, 256),
)

validation_ds = image_dataset_from_directory(
    directory="sports-classification/valid",
    labels="inferred",
    label_mode="int",
    batch_size=64,
    image_size=(256, 256),
)


train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

model = Sequential()
model.add(
    Conv2D(
        128,
        kernel_size=(3, 3),
        padding="valid",
        activation="leaky_relu",
        input_shape=(256, 256, 3),
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(Conv2D(64, kernel_size=(3, 3), padding="valid", activation="leaky_relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(Conv2D(32, kernel_size=(3, 3), padding="valid", activation="leaky_relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(Flatten())
model.add(Dense(512, activation="leaky_relu"))
model.add(Dropout(0.1))
model.add(Dense(256, activation="leaky_relu"))
model.add(Dropout(0.1))
model.add(Dense(128, activation="leaky_relu"))
model.add(Dropout(0.1))
model.add(Dense(100, activation="softmax"))


model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2)
history = model.fit(
    train_ds,
    epochs=50,
    batch_size=32,
    callbacks=[callback],
    validation_data=validation_ds,
)

plot_loss(history)
plot_accuracy(history)
