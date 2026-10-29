import json
import hashlib
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)

st.set_page_config(
    page_title="AI Fruit Scanner | Pisang Ambon",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "banana_model.keras"
LABEL_PATH = BASE_DIR / "models" / "class_names.json"
IMAGE_SIZE = (224, 224)

# FITUR BARU: keranjang pisang disimpan sementara selama sesi Streamlit.
for basket_key in ("basket_ripe", "basket_unripe", "basket_rotten"):
    if basket_key not in st.session_state:
        st.session_state[basket_key] = 0

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --green: #078653;
        --green-dark: #064e3b;
        --green-light: #e9f8f0;
        --yellow: #eab308;
        --red: #ef4444;
        --text: #142b23;
        --muted: #5d7065;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f5f8f6;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(245,248,246,0.96);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #062c23, #031d18);
    }

    section[data-testid="stSidebar"] * {
        color: #f0fdf4;
    }

    .brand {
        padding: 8px 0 18px;
    }

    .brand-icon {
        font-size: 42px;
    }

    .brand h2 {
        color: #ffffff !important;
        font-size: 23px;
        font-weight: 800;
        margin: 4px 0;
    }

    .brand p {
        color: #c7e3d5 !important;
        font-size: 13px;
        line-height: 1.65;
    }

    .sidebar-card {
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 14px;
        padding: 15px;
        margin-top: 14px;
    }

    .sidebar-card-title {
        color: #ffffff;
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .class-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 0;
        font-size: 13px;
        line-height: 1.45;
        color: #f0fdf4;
    }

    .dot {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        display: inline-block;
        flex: 0 0 11px;
    }

    .hero {
        background:
            radial-gradient(circle at 90% 20%, rgba(74,222,128,.23), transparent 30%),
            linear-gradient(115deg, #062f24 0%, #087b4c 100%);
        border-radius: 22px;
        padding: 30px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 12px 32px rgba(6,78,59,.13);
    }

    .pill {
        display: inline-block;
        background: rgba(255,255,255,.15);
        border: 1px solid rgba(255,255,255,.24);
        border-radius: 100px;
        padding: 6px 12px;
        color: #ffffff;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .3px;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: clamp(27px, 3vw, 38px);
        line-height: 1.2;
        font-weight: 800;
        margin: 13px 0;
    }

    .hero p {
        color: #e0f5e8 !important;
        font-size: 14px;
        line-height: 1.75;
        max-width: 680px;
        margin-bottom: 0;
    }

    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3 {
        color: #142b23;
    }

    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label {
        color: #42564b;
    }

    .section-subtitle {
        color: var(--muted) !important;
        font-size: 13px;
        line-height: 1.65;
        margin-bottom: 15px;
    }

    .panel-heading {
        color: #142b23;
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .panel-description {
        color: var(--muted) !important;
        font-size: 12px;
        line-height: 1.6;
        margin-bottom: 14px;
    }

    .result-hero {
        text-align: center;
        padding: 20px 14px;
        background: linear-gradient(180deg, #effaf3, #ffffff);
        border: 1px solid #dceee3;
        border-radius: 16px;
        margin: 12px 0 16px;
    }

    .result-emoji {
        font-size: 46px;
        margin-bottom: 6px;
    }

    .result-title {
        font-size: 24px;
        font-weight: 800;
        color: #142b23;
        margin: 7px 0;
    }

    .result-subtitle {
        color: #5d7065 !important;
        font-size: 12px;
        line-height: 1.65;
    }

    .metric-box {
        background: #f1f7f3;
        border: 1px solid #e2eee6;
        border-radius: 14px;
        padding: 15px;
        margin: 12px 0 18px;
    }

    .metric-label {
        color: #5d7065;
        font-size: 12px;
    }

    .metric-value {
        color: #064e3b;
        font-size: 29px;
        font-weight: 800;
        margin-top: 4px;
    }

    .feature-card {
        background: #ffffff;
        border: 1px solid #e0e9e3;
        border-radius: 15px;
        padding: 17px;
        min-height: 120px;
        box-shadow: 0 4px 16px rgba(20,43,35,.025);
    }

    .feature-icon {
        font-size: 25px;
    }

    .feature-title {
        color: #142b23;
        font-weight: 800;
        font-size: 14px;
        margin: 7px 0 5px;
    }

    .feature-text {
        color: #5d7065;
        font-size: 12px;
        line-height: 1.65;
    }

    .footer {
        text-align: center;
        color: #718078;
        font-size: 11px;
        padding: 22px 0 5px;
    }

    div.stButton > button {
        border-radius: 11px;
        min-height: 45px;
        font-weight: 700;
        border: none;
        transition: all .2s ease;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #078653, #10a66a);
        color: #ffffff;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #066b43;
        color: #ffffff;
        box-shadow: 0 5px 15px rgba(7,134,83,.18);
    }

    div[data-testid="stFileUploader"],
    div[data-testid="stCameraInput"] {
        background: #ffffff;
        border: 1px dashed #9acbb0;
        border-radius: 13px;
        padding: 10px;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #e5ece7;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: .8rem .9rem 1.5rem;
        }

        .hero {
            padding: 23px 20px;
            border-radius: 17px;
        }

        .hero h1 {
            font-size: 27px;
        }

        .hero p {
            font-size: 13px;
        }

        .result-title {
            font-size: 22px;
        }

        .feature-card {
            min-height: auto;
            margin-bottom: 10px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CATEGORY_INFO = {
    "Ripe": {
        "title": "Pisang Matang",
        "emoji": "🍌",
        "color": "#eab308",
        "description": "Model mendeteksi pisang dalam kategori matang.",
        "tips": "Periksa warna kulit dan tekstur sebelum dikonsumsi.",
    },
    "Unripe": {
        "title": "Pisang Mentah",
        "emoji": "🌱",
        "color": "#16a34a",
        "description": "Model mendeteksi pisang dalam kategori mentah.",
        "tips": "Pisang dapat diperam lebih lanjut sesuai kebutuhan.",
    },
    "Rotten": {
        "title": "Pisang Busuk",
        "emoji": "⚠️",
        "color": "#ef4444",
        "description": "Model mendeteksi ciri visual kategori busuk.",
        "tips": "Periksa kondisi fisik dan aroma secara langsung.",
    },
}

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan:\n{MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


@st.cache_data
def load_labels():
    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"File label tidak ditemukan:\n{LABEL_PATH}"
        )

    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    labels = data.get("labels") if isinstance(data, dict) else data

    if (
        not isinstance(labels, list)
        or len(labels) != 3
        or not all(isinstance(label, str) for label in labels)
    ):
        raise ValueError(
            "class_names.json harus berisi tepat tiga label."
        )

    if set(labels) != {"Unripe", "Ripe", "Rotten"}:
        raise ValueError(
            "Label harus terdiri dari Unripe, Ripe, dan Rotten. "
            "Pastikan urutannya sama dengan output model."
        )

    return labels

@st.cache_resource
def load_object_checker():
    return MobileNetV2(
        weights="imagenet",
        include_top=True,
    )


def check_is_banana(image):
    """
    Pemeriksaan awal menggunakan MobileNetV2 ImageNet.
    Pemeriksaan ini bersifat perkiraan, bukan detektor objek
    pisang yang terjamin akurat.
    """
    checker = load_object_checker()

    img = ImageOps.exif_transpose(image).convert("RGB")
    img = img.resize((224, 224))

    array = np.asarray(img, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)

    predictions = checker.predict(array, verbose=0)
    results = decode_predictions(predictions, top=5)[0]

    banana_score = 0.0

    for _, label, score in results:
        if label.lower() == "banana":
            banana_score = float(score)
            break

    is_banana = banana_score >= 0.10

    return is_banana, results, banana_score

def preprocess_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32)

    return np.expand_dims(array, axis=0)


def predict_image(model, image, class_names):
    array = preprocess_image(image)

    scores = model.predict(array, verbose=0)[0]

    if len(scores) != len(class_names):
        raise ValueError(
            "Jumlah output model tidak sesuai jumlah label."
        )

    index = int(np.argmax(scores))

    return (
        class_names[index],
        float(scores[index]),
        scores,
    )

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🍌</div>
            <h2>AI Fruit Scanner</h2>
            <p>
                Klasifikasi kematangan pisang Ambon
                menggunakan Deep Learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "NAVIGASI",
        [
            "🏠  Beranda",
            "📘  Tentang Aplikasi",
            "💡  Cara Penggunaan",
        ],
        label_visibility="collapsed",
        key="page_navigation",
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">
                🍌 Kategori Klasifikasi
            </div>
            <div class="class-row">
                <span class="dot" style="background:#16a34a;"></span>
                <span>Unripe - Pisang mentah</span>
            </div>
            <div class="class-row">
                <span class="dot" style="background:#eab308;"></span>
                <span>Ripe - Pisang matang</span>
            </div>
            <div class="class-row">
                <span class="dot" style="background:#ef4444;"></span>
                <span>Rotten - Pisang busuk</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
        <span class="pill">
            ✦ POWERED BY ARTIFICIAL INTELLIGENCE
        </span>
        <h1>Kenali Kematangan<br>Pisang Ambon dengan AI.</h1>
        <p>
            Unggah foto atau gunakan kamera untuk membantu model
            MobileNetV2 mengklasifikasikan pisang mentah, matang,
            atau busuk.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "📘  Tentang Aplikasi":
    st.header("Tentang AI Fruit Scanner")

    st.write(
        "AI Fruit Scanner adalah aplikasi prototipe klasifikasi "
        "gambar pisang Ambon menggunakan transfer learning "
        "dengan arsitektur MobileNetV2."
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">Deep Learning</div>
                <div class="feature-text">
                    Menggunakan model MobileNetV2.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🖼️</div>
                <div class="feature-title">Klasifikasi Citra</div>
                <div class="feature-text">
                    Menganalisis gambar dengan input 224 × 224.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Tiga Kategori</div>
                <div class="feature-text">
                    Unripe, Ripe, dan Rotten.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Masukkan akurasi aktual berdasarkan evaluasi dataset "
        "pengujian. Akurasi pada dataset tidak menjamin hasil "
        "yang sama pada gambar baru."
    )

    st.warning(
        "Prediksi visual tidak dapat memastikan keamanan buah "
        "secara pasti."
    )

elif page == "💡  Cara Penggunaan":
    st.header("Cara Menggunakan Aplikasi")

    st.markdown(
        """
        ### Mode Upload Foto

        1. Pilih menu **Beranda**.
        2. Pilih **Upload Foto**.
        3. Unggah gambar berformat JPG, JPEG, PNG, atau WEBP.
        4. Tekan **Analisis Gambar**.
        5. Lihat kategori prediksi dan skor tiap kelas.

        ### Mode Kamera

        1. Pilih menu **Beranda**.
        2. Pilih **Mode Kamera**.
        3. Izinkan akses kamera pada browser.
        4. Ambil foto pisang.
        5. Tekan **Analisis Gambar**.

        ### Tips Foto yang Baik

        - Pastikan pisang terlihat jelas.
        - Gunakan pencahayaan yang cukup.
        - Hindari gambar buram atau terlalu gelap.
        - Usahakan pisang menjadi objek utama foto.
        """
    )

    st.warning(
        "Hasil klasifikasi merupakan prediksi visual. Periksa "
        "kondisi buah secara langsung sebelum memutuskan "
        "untuk mengonsumsinya."
    )

else:
    try:
        model = load_model()
        class_names = load_labels()

        if model.output_shape[-1] != len(class_names):
            raise ValueError(
                "Output model dan jumlah label tidak cocok."
            )

    except Exception as error:
        st.error("Model atau label gagal dimuat.")
        st.exception(error)
        st.stop()

    status_left, status_right = st.columns([3, 1])

    with status_left:
        st.header("Mulai Analisis")

        st.markdown(
            """
            <div class="section-subtitle">
                Pilih gambar dari perangkat atau ambil foto
                melalui kamera.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with status_right:
        st.success("● Model siap digunakan")

    source_mode = st.radio(
        "Pilih sumber gambar",
        ["📁 Upload Foto", "📷 Mode Kamera"],
        horizontal=True,
        key="source_mode",
    )

    upload_col, result_col = st.columns([1, 1], gap="large")

    uploaded_file = None
    current_image = None
    image_key = None

    with upload_col:
        st.markdown(
            '<div class="panel-heading">📁 Input Gambar</div>',
            unsafe_allow_html=True,
        )

        if source_mode == "📁 Upload Foto":
            st.markdown(
                """
                <div class="panel-description">
                    Pilih foto pisang Ambon dari perangkat kamu.
                </div>
                """,
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "Pilih gambar pisang",
                type=["jpg", "jpeg", "png", "webp"],
                help="Gunakan gambar yang jelas dan tidak buram.",
                key="upload_banana",
            )

        else:
            st.markdown(
                """
                <div class="panel-description">
                    Izinkan akses kamera, arahkan kamera ke pisang,
                    lalu ambil foto.
                </div>
                """,
                unsafe_allow_html=True,
            )

            uploaded_file = st.camera_input(
                "Ambil foto pisang Ambon",
                key="camera_banana",
            )

        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.getvalue()
                image_key = hashlib.sha256(raw_bytes).hexdigest()

                with Image.open(uploaded_file) as opened_image:
                    current_image = ImageOps.exif_transpose(
                        opened_image
                    ).convert("RGB")

                st.image(
                    current_image,
                    caption="Gambar siap dianalisis",
                    width="stretch",
                )

                st.caption(
                    f"Resolusi: {current_image.width} × "
                    f"{current_image.height} piksel"
                )

            except Exception as error:
                current_image = None
                image_key = None

                st.error("Gambar tidak dapat dibaca.")
                st.caption(str(error))

        elif source_mode == "📁 Upload Foto":
            st.caption(
                "Pilih gambar menggunakan tombol upload di atas."
            )

        else:
            st.info(
                "Ambil foto menggunakan kontrol kamera di atas."
            )

        analyze_clicked = st.button(
            "✦  Analisis Gambar",
            type="primary",
            width="stretch",
            disabled=current_image is None,
            key="analyze_banana",
        )

    current_result_key = (
        f"{source_mode}:{image_key}"
        if image_key is not None
        else None
    )

    if (
        st.session_state.get("result_image_key")
        != current_result_key
    ):
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("object_check_result", None)
        st.session_state.pop("execution_metrics", None)

        st.session_state["result_image_key"] = current_result_key

    if analyze_clicked and current_image is not None:
        start_time = time.perf_counter()

        object_check_time = 0.0
        classification_time = 0.0

        st.session_state.pop("prediction_result", None)
        st.session_state.pop("object_check_result", None)
        st.session_state.pop("execution_metrics", None)

        try:
            object_start = time.perf_counter()

            with st.spinner(
                "Memeriksa apakah gambar berisi pisang..."
            ):
                (
                    is_banana,
                    object_results,
                    banana_score,
                ) = check_is_banana(current_image)

            object_check_time = (
                time.perf_counter() - object_start
            )

            st.session_state["object_check_result"] = {
                "is_banana": is_banana,
                "banana_score": banana_score,
                "object_results": [
                    (label, float(score))
                    for _, label, score in object_results
                ],
            }

            if is_banana:
                classification_start = time.perf_counter()

                with st.spinner(
                    "Mengklasifikasikan kematangan pisang..."
                ):
                    (
                        predicted_class,
                        confidence,
                        scores,
                    ) = predict_image(
                        model,
                        current_image,
                        class_names,
                    )

                classification_time = (
                    time.perf_counter() - classification_start
                )

                st.session_state["prediction_result"] = {
                    "class": predicted_class,
                    "confidence": confidence,
                    "scores": scores.tolist(),
                }

            total_time = time.perf_counter() - start_time

            status = "SUCCESS" if is_banana else "REJECT"

            st.session_state["execution_metrics"] = {
                "object_check_time": object_check_time,
                "classification_time": classification_time,
                "total_time": total_time,
                "status": status,
            }

            st.session_state["result_image_key"] = (
                current_result_key
            )

            print(f"\n[AI FRUIT SCANNER - {status}]")
            print(
                f"Object check    : {object_check_time:.4f} seconds"
            )
            print(
                f"Classification  : "
                f"{classification_time:.4f} seconds"
            )
            print(
                f"Total execution : {total_time:.4f} seconds"
            )
            print("-" * 45)

        except Exception as error:
            total_time = time.perf_counter() - start_time

            st.session_state.pop("prediction_result", None)
            st.session_state.pop("object_check_result", None)

            st.session_state["execution_metrics"] = {
                "object_check_time": object_check_time,
                "classification_time": classification_time,
                "total_time": total_time,
                "status": "ERROR",
            }

            print("\n[AI FRUIT SCANNER - ERROR]")
            print(f"Total execution: {total_time:.4f} seconds")
            print(f"Error: {error}")
            print("-" * 45)

            st.error("Pemeriksaan atau prediksi gagal dilakukan.")
            st.exception(error)

    with result_col:
        st.markdown(
            '<div class="panel-heading">🧠 Hasil Klasifikasi</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="panel-description">
                Pemeriksaan objek dilanjutkan dengan analisis
                kematangan.
            </div>
            """,
            unsafe_allow_html=True,
        )

        object_check = st.session_state.get(
            "object_check_result"
        )

        result = st.session_state.get("prediction_result")

        execution_metrics = st.session_state.get(
            "execution_metrics"
        )

        result_is_current = (
            current_result_key is not None
            and st.session_state.get("result_image_key")
            == current_result_key
        )

        if result_is_current and execution_metrics is not None:
            st.markdown("#### ⏱️ Performa Analisis")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "Pemeriksaan Objek",
                    f"{execution_metrics['object_check_time']:.3f} s",
                )

            with metric2:
                st.metric(
                    "Klasifikasi",
                    f"{execution_metrics['classification_time']:.3f} s",
                )

            with metric3:
                st.metric(
                    "Total Waktu",
                    f"{execution_metrics['total_time']:.3f} s",
                )

        if not result_is_current:
            st.markdown(
                """
                <div class="result-hero">
                    <div class="result-emoji">🍌</div>
                    <div class="result-title">Menunggu Analisis</div>
                    <div class="result-subtitle">
                        Unggah gambar atau ambil foto, lalu tekan
                        Analisis Gambar.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.info(
                "Hasil pemeriksaan dan prediksi akan muncul di area ini."
            )

        elif object_check is not None and not object_check["is_banana"]:
            st.error(
                "🚫 Gambar tidak berhasil dikenali sebagai pisang. "
                "Coba gunakan foto pisang Ambon yang terlihat jelas."
            )

            st.caption(
                "Skor label banana dari pemeriksa objek: "
                f"{object_check['banana_score']:.2%}"
            )

            with st.expander("Detail pemeriksaan objek"):
                st.write("Lima prediksi ImageNet teratas:")

                for label, score in object_check["object_results"]:
                    st.write(
                        f"- {label.replace('_', ' ')}: {score:.2%}"
                    )

            st.warning(
                "Pemeriksaan ini menggunakan model pralatih umum. "
                "Model dapat menolak foto pisang yang benar atau "
                "menerima objek lain. Skor rendah bukan bukti "
                "bahwa gambar pasti bukan pisang."
            )

        elif result is None:
            st.markdown(
                """
                <div class="result-hero">
                    <div class="result-emoji">🔎</div>
                    <div class="result-title">Belum Ada Hasil</div>
                    <div class="result-subtitle">
                        Tekan Analisis Gambar untuk memulai pemeriksaan.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.success(
                "🍌 Objek lolos pemeriksaan sebagai kemungkinan pisang."
            )

            predicted_class = result["class"]
            confidence = float(result["confidence"])
            scores = result["scores"]

            info = CATEGORY_INFO.get(
                predicted_class,
                {
                    "title": predicted_class,
                    "emoji": "🍌",
                    "color": "#078653",
                    "description": "Hasil prediksi model.",
                    "tips": "Periksa kondisi buah secara langsung.",
                },
            )

            st.markdown(
                f"""
                <div class="result-hero">
                    <div class="result-emoji">{info["emoji"]}</div>
                    <div style="font-size:11px;color:{info["color"]};
                                font-weight:800;letter-spacing:1px">
                        {predicted_class.upper()}
                    </div>
                    <div class="result-title">{info["title"]}</div>
                    <div class="result-subtitle">
                        {info["description"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">
                        Skor keyakinan prediksi
                    </div>
                    <div class="metric-value">
                        {confidence * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if confidence < 0.70:
                st.warning(
                    "Skor prediksi relatif rendah. Coba gunakan foto "
                    "yang lebih jelas dengan pencahayaan yang baik."
                )

            st.markdown("**Perbandingan skor setiap kelas**")

            for index, label in enumerate(class_names):
                score = float(scores[index])

                color = CATEGORY_INFO.get(
                    label, {}
                ).get("color", "#078653")

                bar_width = max(
                    0.0,
                    min(score * 100, 100.0),
                )

                st.markdown(
                    f"""
                    <div style="display:flex;
                                justify-content:space-between;
                                gap:10px;font-size:13px;
                                margin-top:12px">
                        <span>{label}</span>
                        <b>{score * 100:.2f}%</b>
                    </div>
                    <div style="background:#e8eeea;
                                height:9px;border-radius:20px;
                                overflow:hidden;margin-top:6px">
                        <div style="width:{bar_width:.4f}%;
                                    background:{color};height:9px;
                                    border-radius:20px">
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.info(
                info["tips"]
                + " Hasil ini merupakan prediksi visual, "
                "bukan jaminan kondisi buah."
            )

    # ==========================================
    # KERANJANG PISANG
    # ==========================================
    st.divider()
    st.header("🧺 Keranjang Pisang")
    st.markdown(
        '<div class="section-subtitle">'
        'Hasil klasifikasi hanya dihitung setelah kamu menekan ' 
        '<b>Tambahkan ke Keranjang</b>.'
        '</div>',
        unsafe_allow_html=True,
    )

    ripe_count = int(st.session_state["basket_ripe"])
    unripe_count = int(st.session_state["basket_unripe"])
    rotten_count = int(st.session_state["basket_rotten"])
    total_count = ripe_count + unripe_count + rotten_count

    basket_notice = st.session_state.pop("basket_notice", None)
    if basket_notice:
        st.success(basket_notice)

    basket1, basket2, basket3, basket4 = st.columns(4)
    with basket1:
        st.metric("🍌 Ripe Basket", ripe_count, help="Jumlah pisang matang yang dikonfirmasi.")
    with basket2:
        st.metric("🌱 Unripe Basket", unripe_count, help="Jumlah pisang mentah yang dikonfirmasi.")
    with basket3:
        st.metric("⚠️ Rotten Basket", rotten_count, help="Jumlah pisang busuk yang dikonfirmasi.")
    with basket4:
        st.metric("📦 Total Pisang", total_count)

    # Hanya izinkan menambah hasil klasifikasi terbaru yang valid.
    cart_result = st.session_state.get("prediction_result")
    cart_object_check = st.session_state.get("object_check_result")
    cart_result_is_current = (
        current_result_key is not None
        and st.session_state.get("result_image_key") == current_result_key
        and cart_result is not None
        and cart_object_check is not None
        and cart_object_check.get("is_banana", False)
    )

    if cart_result_is_current:
        cart_label = str(cart_result.get("class", "")).strip()
        cart_info = CATEGORY_INFO.get(cart_label)
        if cart_info:
            st.info(f"Hasil terakhir yang siap dikonfirmasi: **{cart_label} — {cart_info['title']}**")
        else:
            st.warning("Label hasil prediksi tidak dikenali. Periksa class_names.json.")
    else:
        st.caption("Lakukan analisis gambar pisang yang berhasil terlebih dahulu untuk mengaktifkan tombol tambah.")

    cart_label = (
        str(cart_result.get("class", "")).strip()
        if cart_result is not None
        else ""
    )
    cart_label_valid = cart_label in CATEGORY_INFO

    add_col, reset_col = st.columns(2)
    with add_col:
        add_to_basket = st.button(
            "➕ Tambahkan ke Keranjang",
            type="primary",
            width="stretch",
            disabled=not cart_result_is_current or not cart_label_valid,
            key="add_banana_to_basket",
        )
    with reset_col:
        reset_basket = st.button(
            "🔄 Reset Keranjang",
            width="stretch",
            key="reset_banana_basket",
        )

    if add_to_basket and cart_result_is_current:
        label_to_add = str(cart_result.get("class", "")).strip()
        basket_key_by_label = {
            "Ripe": "basket_ripe",
            "Unripe": "basket_unripe",
            "Rotten": "basket_rotten",
        }
        basket_key = basket_key_by_label.get(label_to_add)
        if basket_key is not None:
            st.session_state[basket_key] += 1
            st.session_state["basket_notice"] = (
                f"{label_to_add} berhasil ditambahkan ke keranjang."
            )
            st.rerun()

    if reset_basket:
        st.session_state["basket_ripe"] = 0
        st.session_state["basket_unripe"] = 0
        st.session_state["basket_rotten"] = 0
        st.session_state["basket_notice"] = (
            "Semua jumlah keranjang berhasil direset."
        )
        st.rerun()


    st.divider()
    st.subheader("Fitur Aplikasi")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Prediksi Cepat</div>
                <div class="feature-text">
                    Analisis dilakukan setelah tombol ditekan.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📷</div>
                <div class="feature-title">Upload & Kamera</div>
                <div class="feature-text">
                    Unggah gambar atau ambil foto langsung
                    menggunakan kamera.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Tiga Kategori</div>
                <div class="feature-text">
                    Lihat prediksi mentah, matang, atau busuk
                    beserta skornya.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(    
    """
    <div class="footer">
        AI Fruit Scanner · MobileNetV2 · Proyek Kecerdasan Buatan
    </div>
    """,
    unsafe_allow_html=True,
)
