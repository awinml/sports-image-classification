from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Dropout

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import InceptionV3

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

inception = InceptionV3(
    input_shape=(256, 256, 3), include_top=False, weights="imagenet", classes=100
)
inception.trainable = False
model = Sequential()
model.add(inception)
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(256, activation="leaky_relu"))
model.add(Dropout(0.1))
model.add(Dense(128, activation="leaky_relu"))
model.add(Dropout(0.1))
model.add(Dense(100, activation="softmax"))

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

callback = EarlyStopping(monitor="loss", patience=2)
history = model.fit(
    train_ds,
    epochs=50,
    batch_size=32,
    callbacks=[callback],
    validation_data=validation_ds,
)

plot_loss(history)
plot_accuracy(history)
