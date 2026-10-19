import os
import json
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

# ==========================================
# 1. KONFIGURASI
# ==========================================

SEED = 123
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_INITIAL = 12
EPOCHS_FINE_TUNE = 8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "banana_dataset_3class")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "banana_mobilenetv2.keras")
LABEL_PATH = os.path.join(MODEL_DIR, "class_names.json")
TFLITE_PATH = os.path.join(MODEL_DIR, "banana_mobilenetv2.tflite")

tf.keras.utils.set_random_seed(SEED)

# ==========================================
# 2. PERIKSA FOLDER DATASET
# ==========================================

for folder in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder}")

print("Folder dataset ditemukan.")

# ==========================================
# 3. MEMUAT DATASET
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

class_names = train_ds.class_names
print("Urutan kelas:", class_names)

expected_classes = ["Ripe", "Rotten", "Unripe"]
if sorted(class_names) != sorted(expected_classes):
    raise ValueError(
        f"Kelas tidak sesuai. Ditemukan: {class_names}. "
        f"Seharusnya: {expected_classes}"
    )

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Simpan urutan label untuk aplikasi Streamlit
with open(LABEL_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

# ==========================================
# 4. CLASS WEIGHT
# Menangani jumlah gambar antarkelas yang tidak seimbang
# ==========================================

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

print("Class weight:", class_weights)

# Optimasi pipeline input
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
], name="data_augmentation")

# ==========================================
# 6. MEMBANGUN MODEL MOBILENETV2
# ==========================================

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

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 7. CALLBACKS
# ==========================================

callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
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

# ==========================================
# 8. TRAINING TAHAP PERTAMA
# ==========================================

print("\n===== TRAINING TAHAP 1: FEATURE EXTRACTION =====")

model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_INITIAL,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================
# 9. FINE-TUNING MOBILENETV2
# ==========================================

print("\n===== TRAINING TAHAP 2: FINE-TUNING =====")

base_model.trainable = True

# Buka sebagian layer akhir untuk penyesuaian
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Batch normalization tetap dibekukan agar training lebih stabil
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

# Muat checkpoint terbaik
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# 10. EVALUASI DATA TEST
# ==========================================

print("\n===== EVALUASI DATA TEST =====")

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0
))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
))

# ==========================================
# 11. SIMPAN MODEL DAN EKSPOR TFLITE
# ==========================================

model.save(MODEL_PATH)
print(f"\nModel Keras tersimpan: {MODEL_PATH}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Model TFLite tersimpan: {TFLITE_PATH}")
print(f"Label kelas tersimpan: {LABEL_PATH}")
print("\nSELESAI! Training, evaluasi, dan ekspor model berhasil.")

{
  "status_code": 422,
  "status": "error",
  "error_type": "UNPROCESSABLE_IMAGE",
  "message": "Gambar tidak dapat dibaca atau berkas rusak.",
  "details": "PIL.UnidentifiedImageError: cannot identify image file"
}

