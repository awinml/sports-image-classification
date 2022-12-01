from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
from sklearn.metrics import classification_report

test_ds = image_dataset_from_directory(
    directory="sports-classification/test",
    labels="inferred",
    label_mode="int",
    batch_size=64,
    image_size=(256, 256),
)

test_ds = test_ds.map(process)

model = load_model("best_model.h5")

y_pred = np.array([])
y_true = np.array([])
for x, y in test_ds:
    y_pred = np.concatenate([y_pred, model.predict_classes(x)])
    y_true = np.concatenate([y_true, np.argmax(y.numpy(), axis=-1)])

print("Classification Report: \n", classification_report(y_pred, y_true))
