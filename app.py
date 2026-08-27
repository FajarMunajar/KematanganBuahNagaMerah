"""
app.py — Sistem Deteksi Kematangan Buah Naga (Streamlit)
==========================================================
Website deteksi kematangan buah naga (Ripe / Rotten / Unripe)
menggunakan model YOLOv8 (best.pt) hasil training di notebook
YOLOv8_DragonFruit_v1.ipynb.

Alur halaman (sesuai wireframe):
  1. Halaman Utama        -> pilih "Mulai Deteksi Live" / "Upload Gambar"
                              (memunculkan popup peringatan dulu)
  2. Halaman Deteksi Live -> ambil gambar dari kamera, lalu deteksi
  3. Halaman Upload Gambar-> upload file gambar, lalu deteksi
  4. Halaman Hasil Deteksi-> tampilkan gambar + label + tingkat kemiripan

Cara menjalankan:
    1. Taruh file model 'best.pt' di folder yang sama dengan app.py
       (atau ubah MODEL_PATH di bawah).
    2. pip install -r requirements.txt
    3. streamlit run app.py
"""
 
import base64
import os

import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "best.pt"                      
CLASS_NAMES = ["Ripe", "Rotten", "Unripe"]   
CLASS_LABEL_ID = {"Ripe": "🟢 Ripe (Matang)", "Rotten": "🔴 Rotten (Busuk)", "Unripe": "🟡 Unripe (Belum Matang)"}
CONF_THRESHOLD = 0.25
CLASS_SAMPLE_DIR = "assets/class_samples"   # gambar contoh tiap class untuk halaman "Cara Deteksi"
LOGO_PATH = "assets/logo.png"               # logo sidebar (opsional, fallback ke emoji kalau tidak ada)

# Deskripsi tiap class — nama class HARUS sama dengan CLASS_NAMES di atas (dari data.yaml)
CLASS_INFO = {
    "Ripe": {
        "emoji": "🟢",
        "title": "Ripe (Matang)",
        "desc": "Kulit buah berwarna merah/pink cerah merata dan sisik (sirip) berwarna hijau segar. "
                "Buah pada kondisi ini sudah siap dipanen dan dikonsumsi.",
        "sample": f"{CLASS_SAMPLE_DIR}/Ripe.jpg",
    },
    "Unripe": {
        "emoji": "🟡",
        "title": "Unripe (Belum Matang)",
        "desc": "Kulit buah masih didominasi warna hijau, tekstur masih keras. "
                "Buah pada kondisi ini belum siap dipanen.",
        "sample": f"{CLASS_SAMPLE_DIR}/Unripe.jpg",
    },
    "Rotten": {
        "emoji": "🔴",
        "title": "Rotten (Busuk)",
        "desc": "Kulit buah tampak kusam/kecoklatan dengan bercak busuk, tekstur mulai lembek atau mengerut. "
                "Buah pada kondisi ini tidak layak dikonsumsi.",
        "sample": f"{CLASS_SAMPLE_DIR}/Rotten.jpg",
    },
}

st.set_page_config(
    page_title="Sistem Deteksi Kematangan Buah Naga Merah",
    page_icon="🐉",
    layout="centered",
)

# ============================================================
# STYLE (meniru tampilan browser mockup di wireframe)
# ============================================================
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* header TIDAK disembunyikan — di situ tempat tombol buka/tutup sidebar berada */

        .block-container {
            padding-top: 2.5rem;
            max-width: 760px;
        }
        .center-text { text-align: center; }
        .app-title {
            text-align: center;
            font-weight: 800;
            letter-spacing: 0.5px;
            margin-bottom: 0.25rem;
        }
        .app-subtitle {
            text-align: center;
            color: #9aa0a6;
            font-size: 0.95rem;
            line-height: 1.6;
            padding: 0 1.5rem;
        }
        div[data-testid="stButton"] > button {
            border-radius: 999px;
            padding: 0.85rem 1rem;
            font-size: 1.05rem;
            font-weight: 700;
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            background-color: #2563eb;
            border-color: #2563eb;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            max-width: 640px;
            margin: 0 auto;
        }
        .result-badge {
            text-align: center;
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 0.5rem;
        }
        .conf-text {
            text-align: center;
            color: #555;
            margin-bottom: 1rem;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background-color: #f4f6fd;
        }
        .sidebar-logo {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0 1.6rem 0;
        }
        .sidebar-logo-icon {
            font-size: 2.1rem;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            display: flex;
            align-items: center;
            justify-content: center;
            object-fit: cover;
        }
        .sidebar-logo-text {
            font-weight: 800;
            text-align: center;
            font-size: 1rem;
            line-height: 1.3;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            justify-content: flex-start !important;
            text-align: left !important;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.95rem;
            padding: 0.6rem 1rem;
            border: none;
            background-color: transparent;
            box-shadow: none;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button * {
            text-align: left !important;
            justify-content: flex-start !important;
            width: 100%;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
            background-color: #2563eb;
            color: #fff;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background-color: rgba(37, 99, 235, 0.12);
        }

        /* ---------- Cards ---------- */
        .section-heading {
            font-weight: 800;
            font-size: 1.2rem;
            margin: 1.5rem 0 0.7rem 0;
        }
        .step-card, .tip-card, .class-card {
            background-color: rgba(128, 128, 128, 0.07);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 14px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.7rem;
        }
        .step-row {
            display: flex;
            align-items: flex-start;
            gap: 0.7rem;
        }
        .step-number {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            min-width: 28px;
            border-radius: 50%;
            background-color: #2563eb;
            color: #fff;
            font-weight: 700;
            font-size: 0.85rem;
        }
        .step-text { padding-top: 0.15rem; }
        .step-text b { display: block; margin-bottom: 0.1rem; }
        .tip-card { display: flex; align-items: flex-start; gap: 0.6rem; }
        .class-card { text-align: center; }
        .class-card h4 { margin: 0.6rem 0 0.3rem 0; }
        .class-card p { color: #888; font-size: 0.88rem; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "result_image" not in st.session_state:
    st.session_state.result_image = None
if "detections" not in st.session_state:
    st.session_state.detections = []
if "live_photo" not in st.session_state:
    st.session_state.live_photo = None
if "upload_photo" not in st.session_state:
    st.session_state.upload_photo = None


def go_to(page_name: str):
    st.session_state.page = page_name


def reset_detection_state():
    st.session_state.result_image = None
    st.session_state.detections = []
    st.session_state.live_photo = None
    st.session_state.upload_photo = None


# ============================================================
# MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_PATH)


def run_detection(pil_image: Image.Image):
    """Menjalankan YOLOv8 pada satu gambar dan mengembalikan
    (gambar_hasil_anotasi, list_deteksi_terurut_confidence)."""
    model = load_model()
    # PENTING: kirim objek PIL Image langsung (bukan np.array manual).
    # Kalau di-convert manual ke np.array lalu dikirim sebagai numpy,
    # ultralytics akan membacanya sebagai BGR (konvensi OpenCV) padahal
    # datanya RGB -> channel warna R dan B tertukar -> warna buah naga
    # jadi kacau di mata model -> deteksi gagal/meleset. Dengan
    # mengirim PIL Image langsung, ultralytics menangani konversi
    # warnanya dengan benar.
    results = model.predict(
        source=pil_image.convert("RGB"),
        conf=CONF_THRESHOLD,
        imgsz=640,
        verbose=False,
    )
    result = results[0]

    
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[..., ::-1]

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            detections.append({"class_name": cls_name, "confidence": conf})
        detections.sort(key=lambda d: d["confidence"], reverse=True)

    return annotated_rgb, detections


# ============================================================
# POPUP PERINGATAN (sesuai wireframe "Popup_Peringatan")
# ============================================================
@st.dialog("⚠️ Peringatan")
def show_warning_popup(target_page: str):
    st.markdown(
        """
        <p class="center-text">
        Sistem ini hanya digunakan untuk mendeteksi<br>
        tingkat kematangan buah naga merah.<br>
        Silakan gunakan gambar buah naga merah untuk<br>
        mendapatkan hasil deteksi yang sesuai.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("Mengerti", type="primary", use_container_width=True):
            go_to(target_page)
            st.rerun()


# ============================================================
# HALAMAN 1 — HALAMAN UTAMA
# ============================================================
def page_home():
    st.markdown(
        "<h1 class='app-title'>SISTEM DETEKSI KEMATANGAN<br>BUAH NAGA MERAH</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='app-subtitle'>Sistem ini hanya dilatih untuk mendeteksi tingkat kematangan "
        "menggunakan kamera atau upload gambar. Sistem akan menampilkan Buah Naga Merah tersebut "
        "Matang (Ripe), Belum Matang (Unripe) &amp; Busuk (Rotten).</p>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("")

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("🎥  Mulai Deteksi Live", use_container_width=True):
            show_warning_popup("live")
        if st.button("📤  Upload Gambar", use_container_width=True):
            show_warning_popup("upload")


# ============================================================
# HALAMAN 2 — MULAI DETEKSI LIVE
# ============================================================
def page_live():
    st.markdown("<h2 class='app-title'>Mulai Deteksi Live</h2>", unsafe_allow_html=True)
    st.write("")

    with st.container(border=True):
        photo = st.camera_input("Aktifkan kamera lalu ambil gambar buah naga merah", label_visibility="collapsed")
        if photo is not None:
            st.session_state.live_photo = photo

    st.write("")
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        col1, col2 = st.columns(2)
        with col1:
            detect_clicked = st.button(
                "Mulai Deteksi",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.live_photo is None,
            )
        with col2:
            if st.button("Kembali Ke Home", use_container_width=True):
                reset_detection_state()
                go_to("home")
                st.rerun()

    if detect_clicked and st.session_state.live_photo is not None:
        with st.spinner("Mendeteksi tingkat kematangan buah naga merah..."):
            image = Image.open(st.session_state.live_photo)
            annotated, detections = run_detection(image)
            st.session_state.result_image = annotated
            st.session_state.detections = detections
        go_to("result")
        st.rerun()


# ============================================================
# HALAMAN 3 — UPLOAD GAMBAR
# ============================================================
def page_upload():
    with st.container(border=True):
        st.markdown("<h3 class='app-title'>Upload Gambar Buah Naga Merah</h3>", unsafe_allow_html=True)
        st.write("")

        uploaded = st.file_uploader(
            "Choose File", type=["jpg", "jpeg", "png"], label_visibility="visible"
        )
        if uploaded is not None:
            st.session_state.upload_photo = uploaded
            st.image(uploaded, caption="Preview gambar", use_container_width=True)

        st.write("")
        if st.button(
            "Mulai Deteksi",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.upload_photo is None,
        ):
            with st.spinner("Mendeteksi tingkat kematangan buah naga merah..."):
                image = Image.open(st.session_state.upload_photo)
                annotated, detections = run_detection(image)
                st.session_state.result_image = annotated
                st.session_state.detections = detections
            go_to("result")
            st.rerun()

    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("Kembali Halaman Utama", use_container_width=True):
            reset_detection_state()
            go_to("home")
            st.rerun()


# ============================================================
# HALAMAN 4 — HASIL DETEKSI
# ============================================================
def page_result():
    st.markdown(
        "<h2 class='app-title'>Hasil Deteksi Kematangan Buah Naga Merah</h2>",
        unsafe_allow_html=True,
    )
    st.write("")

    if st.session_state.result_image is not None:
        st.image(st.session_state.result_image, use_container_width=True)
    else:
        st.warning("Belum ada gambar yang dideteksi.")

    detections = st.session_state.detections

    if detections:
        top = detections[0]
        label = CLASS_LABEL_ID.get(top["class_name"], top["class_name"])
        st.markdown(f"<p class='result-badge'>Buah Naga: {label}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='conf-text'>tingkat keyakinan  {top['confidence'] * 100:.0f}%</p>",
            unsafe_allow_html=True,
        )

        if len(detections) > 1:
            with st.expander(f"Lihat semua deteksi ({len(detections)})"):
                for i, d in enumerate(detections, start=1):
                    st.write(
                        f"{i}. {CLASS_LABEL_ID.get(d['class_name'], d['class_name'])} "
                        f"— {d['confidence'] * 100:.0f}%"
                    )
    else:
        st.markdown(
            "<p class='result-badge'>Buah naga merah tidak terdeteksi</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p class='conf-text'>Coba gunakan gambar buah naga merah yang lebih jelas.</p>",
            unsafe_allow_html=True,
        )

    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("Kembali Ke Home", use_container_width=True):
            reset_detection_state()
            go_to("home")
            st.rerun()


# ============================================================
# SIDEBAR NAVIGASI
# ============================================================
NAV_ITEMS = [
    ("home", "🏠  Home"),
    ("cara_deteksi", "📖  Cara Deteksi"),
    ("tentang", "ℹ️  Tentang"),
]
HOME_FLOW_PAGES = {"home", "live", "upload", "result"}


def current_nav_key():
    return "home" if st.session_state.page in HOME_FLOW_PAGES else st.session_state.page


def render_sidebar():
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            with open(LOGO_PATH, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="sidebar-logo-icon">'
        else:
            logo_html = '<div class="sidebar-logo-icon">🐉</div>'

        st.markdown(
            f"""
            <div class="sidebar-logo">
                {logo_html}
                <div class="sidebar-logo-text">Deteksi Kematangan<br>Buah Naga Merah</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        active = current_nav_key()
        for key, label in NAV_ITEMS:
            if st.button(
                label,
                key=f"nav_{key}",
                type="primary" if key == active else "secondary",
                use_container_width=True,
            ):
                if key == "home":
                    reset_detection_state()
                go_to(key)
                st.rerun()


def step_card(number: int, title: str, text: str):
    st.markdown(
        f"""
        <div class="step-card">
            <div class="step-row">
                <div class="step-number">{number}</div>
                <div class="step-text"><b>{title}</b>{text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tip_card(text: str):
    st.markdown(f"<div class='tip-card'>✅ <div>{text}</div></div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN — CARA DETEKSI
# ============================================================
def page_cara_deteksi():
    st.markdown("<h2 class='app-title'>📖 Cara Deteksi</h2>", unsafe_allow_html=True)
    st.write("")

    # A. Cara menggunakan sistem
    st.markdown("<div class='section-heading'>Cara Menggunakan Sistem</div>", unsafe_allow_html=True)
    st.write(
        "Pengguna dapat mengunggah gambar buah naga merah (atau mengambil foto langsung lewat kamera) "
        "untuk kemudian dianalisis oleh sistem menggunakan model deteksi objek. Sistem akan "
        "mengenali buah naga merah pada gambar dan menampilkan tingkat kematangannya secara otomatis."
    )

    # B. Langkah-langkah deteksi
    st.markdown("<div class='section-heading'>Langkah-Langkah Deteksi</div>", unsafe_allow_html=True)
    steps = [
        ("Buka halaman Home", ""),
        ("Siapkan gambar buah naga merah", " yang ingin dideteksi."),
        ("Klik tombol Upload Gambar", " (atau Mulai Deteksi Live untuk pakai kamera)."),
        ("Pilih gambar buah naga merah", " dari perangkat kamu."),
        ("Klik tombol Mulai Deteksi", ""),
        ("Tunggu proses klasifikasi selesai", ""),
        ("Lihat hasil deteksi", " berupa class kematangan dan confidence score."),
    ]
    for i, (title, extra) in enumerate(steps, start=1):
        step_card(i, title, extra)

    # C. Tips
    st.markdown("<div class='section-heading'>Tips Agar Hasil Deteksi Lebih Akurat</div>", unsafe_allow_html=True)
    tips = [
        "Gunakan gambar dengan kualitas yang cukup baik.",
        "Pastikan objek buah naga merah terlihat jelas.",
        "Hindari gambar yang terlalu gelap atau terlalu terang.",
        "Usahakan buah naga merah tidak tertutup oleh objek lain.",
        "Gunakan posisi buah naga merah yang mudah terlihat.",
        "Hindari gambar yang terlalu buram.",
        "Pastikan objek berada dalam area gambar dengan jelas.",
        "Gunakan gambar dengan kondisi yang mirip dengan data yang dipakai untuk melatih model.",
    ]
    for t in tips:
        tip_card(t)

    # D. Class yang dapat dideteksi
    st.markdown("<div class='section-heading'>Class yang Dapat Dideteksi</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for col, cls in zip(cols, CLASS_NAMES):
        info = CLASS_INFO[cls]
        with col:
            with st.container(border=True):
                if os.path.exists(info["sample"]):
                    st.image(info["sample"], use_container_width=True)
                st.markdown(
                    f"<div class='class-card'><h4>{info['emoji']} {info['title']}</h4>"
                    f"<p>{info['desc']}</p></div>",
                    unsafe_allow_html=True,
                )


# ============================================================
# HALAMAN — TENTANG
# ============================================================
def page_tentang():
    st.markdown("<h2 class='app-title'>ℹ️ Tentang</h2>", unsafe_allow_html=True)
    st.write("")

    # A. Tentang sistem
    st.markdown("<div class='section-heading'>Tentang Sistem</div>", unsafe_allow_html=True)
    st.write(
        "Sistem Deteksi Kematangan Buah Naga adalah aplikasi berbasis **computer vision** dan "
        "**machine learning** yang digunakan untuk membantu mendeteksi dan mengklasifikasikan "
        "tingkat kematangan buah naga merah berdasarkan gambar. Tujuan sistem ini adalah membantu "
        "pengguna mengetahui tingkat kematangan buah naga merah secara lebih cepat dan praktis hanya "
        "melalui citra digital, tanpa perlu memeriksa buah secara manual satu per satu."
    )

    # B. Cara kerja sistem
    st.markdown("<div class='section-heading'>Cara Kerja Sistem</div>", unsafe_allow_html=True)
    flow = [
        ("Input Gambar", "Pengguna mengunggah gambar atau mengambil foto buah naga merah lewat kamera."),
        ("Preprocessing", "Gambar diproses dan disesuaikan ukurannya dengan input yang dibutuhkan model."),
        ("Model", "Gambar dianalisis menggunakan model deep learning YOLOv8 yang telah dilatih "
                   "menggunakan dataset buah naga merah."),
        ("Klasifikasi", "Model menentukan class kematangan buah naga merah berdasarkan karakteristik "
                         "visual pada gambar."),
        ("Confidence Score", "Sistem menampilkan tingkat keyakinan model terhadap hasil prediksi "
                              ),
        ("Hasil Deteksi", "Sistem menampilkan class kematangan yang diprediksi model kepada pengguna."),
    ]
    for i, (title, desc) in enumerate(flow, start=1):
        step_card(i, title, f" — {desc}")

    # C. Teknologi yang digunakan
    st.markdown("<div class='section-heading'>Teknologi yang Digunakan</div>", unsafe_allow_html=True)
    tech = [
        ("🐍 Python", "Bahasa pemrograman utama sistem."),
        ("🎈 Streamlit", "Framework untuk membangun antarmuka web aplikasi ini."),
        ("🧠 YOLOv8 (Ultralytics)", "Arsitektur deep learning untuk deteksi & klasifikasi objek."),
        ("🖼️ Pillow (PIL)", "Library untuk membaca dan memproses gambar."),
    ]
    for name, desc in tech:
        st.markdown(
            f"<div class='step-card'><b>{name}</b><br><span style='color:#888;'>{desc}</span></div>",
            unsafe_allow_html=True,
        )


# ============================================================
# ROUTER
# ============================================================
PAGES = {
    "home": page_home,
    "live": page_live,
    "upload": page_upload,
    "result": page_result,
    "cara_deteksi": page_cara_deteksi,
    "tentang": page_tentang,
}

render_sidebar()
PAGES[st.session_state.page]()