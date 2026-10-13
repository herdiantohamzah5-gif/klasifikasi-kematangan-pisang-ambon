import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# 1. Konfigurasi Path (Gunakan folder test dari dataset 5 kelas yang sama!)
MODEL_PATH = "banana_mobilenetv2_model.keras"
DATASET_TEST_DIR = r"C:\Users\User\Downloads\pisang ambon.v1i.folder\test"  # Pastikan path ini menunjuk ke dataset 5 kelas

# 2. Load Dataset Test dengan Preprocessing MobileNetV2
print("Memuat dataset pengujian (test)...")
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    DATASET_TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Wajib False agar urutan data tidak acak
)

# 3. Load Model
print("\nMemuat model...")
model = tf.keras.models.load_model(MODEL_PATH)

# 4. Prediksi Seluruh Data Test
print("Melakukan prediksi pada seluruh data test...")
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# 5. Cetak Classification Report
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_true, y_pred, target_names=class_names))

# 6. Buat dan Tampilkan Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix - Klasifikasi Kematangan Pisang')
plt.ylabel('Kategori Sebenarnya (True Class)')
plt.xlabel('Hasil Prediksi Model (Predicted Class)')
plt.tight_layout()

plt.savefig("confusion_matrix.png")
print("\nGambar Confusion Matrix telah disimpan sebagai 'confusion_matrix.png'")
plt.show()