
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import json
import os

# ==================================================
# 1. KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Klasifikasi Kematangan Pisang Ambon",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. MEMUAT MODEL DAN LABEL KELAS
# ==================================================
MODEL_PATH = "banana_mobilenetv2_model.keras"
LABEL_PATH = "banana_class_names.json"

# Urutan fallback ini hanya benar jika sesuai dengan
# urutan kelas saat model dilatih.
FALLBACK_CLASS_NAMES = [
    "Half_Ripe",
    "Nearly_Rotten",
    "Overripe",
    "Ripe",
    "Unripe"
]


@st.cache_resource
def load_banana_model():
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"File model tidak ditemukan: {MODEL_PATH}"
        )

    try:
        loaded_model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        return loaded_model, None

    except Exception as e:
        return None, f"Gagal memuat model: {e}"


def load_class_names():
    if not os.path.exists(LABEL_PATH):
        return FALLBACK_CLASS_NAMES, (
            "File label tidak ditemukan. Urutan label fallback "
            "belum tentu sesuai dengan urutan training."
        )

    try:
        with open(LABEL_PATH, "r", encoding="utf-8") as f:
            names = json.load(f)

        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise ValueError(
                "Format file label harus berupa daftar nama kelas."
            )

        return names, None

    except Exception as e:
        return None, f"Gagal membaca file label: {e}"


model, model_error = load_banana_model()
CLASS_NAMES, label_warning = load_class_names()

if model_error:
    st.error(model_error)

if label_warning:
    st.warning(label_warning)

# Pastikan jumlah output model dan jumlah label cocok
if model is not None and CLASS_NAMES is not None:
    output_classes = model.output_shape[-1]

    if output_classes != len(CLASS_NAMES):
        st.error(
            f"Jumlah output model ({output_classes}) tidak cocok "
            f"dengan jumlah label ({len(CLASS_NAMES)}). "
            "Periksa model dan file label hasil training."
        )
        model = None

# ==================================================
# 3. INFORMASI SETIAP KELAS
# ==================================================
CLASS_INFO = {
    "Unripe": {
        "title": "Belum Matang (Unripe)",
        "color": "#2E7D32",
        "desc": (
            "Kulit pisang masih hijau dominan. "
            "Tekstur relatif keras dan rasa belum manis."
        )
    },
    "Half_Ripe": {
        "title": "Setengah Matang (Half Ripe)",
        "color": "#FBC02D",
        "desc": (
            "Pisang mulai menguning, tetapi masih terdapat "
            "bagian berwarna hijau."
        )
    },
    "Ripe": {
        "title": "Matang Sempurna (Ripe)",
        "color": "#F57F17",
        "desc": (
            "Kulit umumnya kuning. Pisang biasanya lebih lembut "
            "dan memiliki rasa manis."
        )
    },
    "Overripe": {
        "title": "Terlalu Matang (Overripe)",
        "color": "#E65100",
        "desc": (
            "Pisang dapat memiliki banyak bintik cokelat "
            "dan tekstur lebih lembut."
        )
    },
    "Nearly_Rotten": {
        "title": "Hampir Busuk (Nearly Rotten)",
        "color": "#4E342E",
        "desc": (
            "Pisang dapat menunjukkan perubahan warna gelap "
            "atau kerusakan yang terlihat. Periksa kondisi fisik "
            "karena prediksi gambar tidak memastikan keamanan pangan."
        )
    }
}

# ==================================================
# 4. FUNGSI PREDIKSI
# ==================================================
def predict_image(image_input, model_tf):
    # Gambar dari PIL dikonversi menjadi RGB
    img = ImageOps.exif_transpose(image_input).convert("RGB")
    img = img.resize((224, 224))

    # Nilai piksel tetap 0-255 di sini.
    # Model training sebelumnya memiliki Rescaling ke -1 sampai 1.
    img_array = np.asarray(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model_tf.predict(
        img_array,
        verbose=0
    )

    scores = predictions[0]

    if len(scores) != len(CLASS_NAMES):
        raise ValueError(
            "Jumlah skor prediksi tidak sama dengan jumlah kelas."
        )

    predicted_index = int(np.argmax(scores))
    predicted_name = CLASS_NAMES[predicted_index]
    confidence = float(scores[predicted_index]) * 100

    return predicted_name, confidence, scores


# ==================================================
# 5. TAMPILAN APLIKASI
# ==================================================
st.title("🍌 Deteksi & Klasifikasi Kematangan Pisang Ambon")

st.markdown(
    "Aplikasi berbasis **Deep Learning MobileNetV2** "
    "untuk mengklasifikasikan tingkat kematangan pisang Ambon."
)

st.divider()

st.sidebar.header("📌 Informasi Aplikasi")
st.sidebar.markdown("**Daftar kelas kematangan:**")

if CLASS_NAMES:
    for i, name in enumerate(CLASS_NAMES):
        info = CLASS_INFO.get(name, {})
        title = info.get("title", name)
        st.sidebar.write(f"{i}. {title}")
else:
    st.sidebar.error("Label kelas belum dapat dibaca.")

input_mode = st.radio(
    "Pilih Metode Input Gambar:",
    [
        "📤 Upload File Foto",
        "📸 Ambil Foto via Kamera"
    ],
    horizontal=True
)

source_image = None

if input_mode == "📤 Upload File Foto":
    uploaded_file = st.file_uploader(
        "Unggah foto pisang Ambon (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        source_image = uploaded_file

else:
    camera_file = st.camera_input(
        "Ambil foto pisang Ambon langsung dari kamera"
    )

    if camera_file is not None:
        source_image = camera_file

# ==================================================
# 6. ANALISIS GAMBAR DAN HASIL PREDIKSI
# ==================================================
col1, col2 = st.columns([1, 1])

if source_image is not None:
    image = Image.open(source_image)
    image = ImageOps.exif_transpose(image).convert("RGB")

    with col1:
        st.subheader("📷 Gambar yang Dianalisis")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📊 Hasil Analisis")

        if model is None or CLASS_NAMES is None:
            st.warning(
                "Model atau label kelas belum siap. "
                "Periksa pesan kesalahan di atas."
            )

        else:
            try:
                with st.spinner("Menganalisis kematangan pisang..."):
                    label, confidence, all_scores = predict_image(
                        image,
                        model
                    )

                info = CLASS_INFO.get(label, {})
                title = info.get("title", label)
                color = info.get("color", "#333333")
                description = info.get("desc", "-")

                st.markdown(
                    f"""
                    <div style="
                        background-color: {color};
                        padding: 18px;
                        border-radius: 12px;
                        color: white;
                    ">
                        <h2 style="margin:0;color:white;">
                            {title}
                        </h2>
                        <h3 style="margin-top:12px;color:white;">
                            Tingkat Kepercayaan: {confidence:.2f}%
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")
                st.write(f"**Karakteristik:** {description}")

                st.caption(
                    "Confidence adalah skor keluaran model, "
                    "bukan jaminan bahwa prediksi selalu benar."
                )

                st.divider()

                # Grafik seluruh probabilitas
                st.subheader("📈 Distribusi Probabilitas")

                chart_data = {
                    CLASS_NAMES[i]: float(all_scores[i] * 100)
                    for i in range(len(CLASS_NAMES))
                }

                st.bar_chart(chart_data)

                # Detail untuk pemeriksaan model
                with st.expander(
                    "🔎 Detail prediksi semua kelas"
                ):
                    sorted_indices = np.argsort(all_scores)[::-1]

                    for idx in sorted_indices:
                        class_name = CLASS_NAMES[int(idx)]
                        score = float(all_scores[idx])

                        st.write(
                            f"**{class_name}**: {score * 100:.2f}%"
                        )
                        st.progress(
                            float(np.clip(score, 0.0, 1.0))
                        )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

else:
    st.info(
        "💡 Silakan unggah gambar atau gunakan kamera "
        "untuk mengambil foto pisang."
    )
