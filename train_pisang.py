import os
import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

SEED = 123
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_INITIAL = 12
EPOCHS_FINE_TUNE = 8

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_banana"
MODEL_DIR = BASE_DIR / "models"

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = MODEL_DIR / "banana_model.keras"
WEIGHTS_PATH = MODEL_DIR / "banana_model.weights.h5"
LABEL_PATH = MODEL_DIR / "class_names.json"
TFLITE_PATH = MODEL_DIR / "banana_mobilenetv2.tflite"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"

QUARANTINE_DIR = BASE_DIR / "_quarantine_corrupt_images"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

for folder in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {folder}"
        )

print("Semua folder dataset ditemukan.")


def check_and_quarantine_images():
    corrupted_count = 0

    for split_dir in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
        for file_path in split_dir.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                with Image.open(file_path) as img:
                    img.verify()

                with Image.open(file_path) as img:
                    img.load()

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                SyntaxError
            ) as error:
                relative_path = file_path.relative_to(DATASET_DIR)
                destination = QUARANTINE_DIR / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)

                if destination.exists():
                    destination = destination.with_name(
                        f"{destination.stem}_duplicate"
                        f"{destination.suffix}"
                    )

                shutil.move(str(file_path), str(destination))
                corrupted_count += 1

                print("\nGambar bermasalah dipindahkan:")
                print(f"  Asal   : {file_path}")
                print(f"  Tujuan : {destination}")
                print(f"  Error  : {error}")

    print("\n===== PEMERIKSAAN GAMBAR SELESAI =====")
    print(f"Jumlah gambar bermasalah: {corrupted_count}")

    if corrupted_count:
        print(
            "Gambar bermasalah disimpan di folder "
            f"{QUARANTINE_DIR}"
        )


check_and_quarantine_images()


# ==============================
# MEMUAT DATASET
# ==============================

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

class_names = train_ds.class_names

print("\nUrutan kelas:", class_names)

expected_classes = ["Ripe", "Rotten", "Unripe"]

if sorted(class_names) != sorted(expected_classes):
    raise ValueError(
        f"Kelas tidak sesuai. Ditemukan: {class_names}. "
        f"Seharusnya: {expected_classes}"
    )

valid_ds = tf.keras.utils.image_dataset_from_directory(
    str(VALID_DIR),
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    str(TEST_DIR),
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ==============================
# MENYIMPAN NAMA KELAS
# ==============================

with open(LABEL_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4, ensure_ascii=False)


# ==============================
# CLASS WEIGHT
# ==============================

train_labels = np.concatenate([
    labels.numpy() for _, labels in train_ds
])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=train_labels
)

class_weights = {
    int(index): float(weight)
    for index, weight in enumerate(weights)
}

print("\nClass weight:", class_weights)


# ==============================
# OPTIMASI DATASET
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ==============================
# DATA AUGMENTATION
# ==============================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10)
    ],
    name="data_augmentation"
)


# ==============================
# MEMBANGUN MODEL MOBILENETV2
# ==============================

base_model = MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(*IMAGE_SIZE, 3))

x = data_augmentation(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==============================
# CALLBACK TRAINING
# ==============================

callbacks = [
    ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


# ==============================
# TRAINING TAHAP 1
# ==============================

print("\n===== TRAINING TAHAP 1: FEATURE EXTRACTION =====")

model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_INITIAL,
    class_weight=class_weights,
    callbacks=callbacks
)


# ==============================
# TRAINING TAHAP 2
# ==============================

print("\n===== TRAINING TAHAP 2: FINE-TUNING =====")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_FINE_TUNE,
    class_weight=class_weights,
    callbacks=callbacks
)


# ==============================
# MEMUAT MODEL TERBAIK
# ==============================

print("\n===== MEMUAT MODEL TERBAIK =====")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Checkpoint model tidak ditemukan. "
        "Periksa hasil training dan error di atas."
    )

model = tf.keras.models.load_model(str(MODEL_PATH))

print(f"Checkpoint berhasil dimuat: {MODEL_PATH}")


# ==============================
# EVALUASI DATA TEST
# ==============================

print("\n===== EVALUASI DATA TEST =====")

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ==============================
# PREDIKSI DATA TEST
# ==============================

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)

y_pred = np.argmax(probabilities, axis=1)


# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ==============================
# CONFUSION MATRIX ANGKA
# ==============================

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

print(cm)


# ==============================
# VISUALISASI CONFUSION MATRIX
# ==============================

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)

plt.title("Confusion Matrix - AI Fruit Scanner")
plt.colorbar()

tick_marks = np.arange(len(class_names))

plt.xticks(
    tick_marks,
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    class_names
)

threshold = cm.max() / 2 if cm.size > 0 else 0

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )

plt.xlabel("Prediksi Model")
plt.ylabel("Label Sebenarnya")
plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Gambar confusion matrix tersimpan: "
    f"{CONFUSION_MATRIX_PATH}"
)


# ==============================
# MENYIMPAN BOBOT MODEL
# ==============================

model.save_weights(str(WEIGHTS_PATH))

print(f"\nBobot model tersimpan: {WEIGHTS_PATH}")
print(f"Model lengkap tersimpan: {MODEL_PATH}")


# ==============================
# EKSPOR MODEL TFLITE
# ==============================

print("\n===== EKSPOR MODEL TFLITE =====")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as file:
    file.write(tflite_model)

print(f"Model TFLite tersimpan: {TFLITE_PATH}")
print(f"Label kelas tersimpan: {LABEL_PATH}")


# ==============================
# SELESAI
# ==============================

print("\nSELESAI!")
print("Training, evaluasi, dan ekspor model berhasil.")
