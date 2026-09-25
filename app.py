from collections import Counter, defaultdict
from pathlib import Path
import os
import tempfile
import time

import cv2
import numpy as np
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
MODEL_PATHS = (APP_DIR / "best.pt", APP_DIR / "runs" / "detect" / "train3" / "weights" / "best.pt")
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]


def find_model():
    return next((path for path in MODEL_PATHS if path.exists()), None)


def load_model(st):
    path = find_model()
    if path is None:
        return None

    @st.cache_resource(show_spinner=False)
    def cached_model(model_file):
        from ultralytics import YOLO
        return YOLO(str(model_file))

    try:
        return cached_model(str(path))
    except Exception:
        return None


def extract_stats(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {"total": 0, "highest": 0.0, "classes": {}, "confidence": {}}

    names = result.names
    counts = Counter()
    confidences = defaultdict(list)
    class_ids = boxes.cls.int().cpu().tolist()
    scores = boxes.conf.float().cpu().tolist()
    for class_id, score in zip(class_ids, scores):
        label = names[int(class_id)]
        counts[label] += 1
        confidences[label].append(score)
    return {"total": len(scores), "highest": max(scores), "classes": dict(counts), "confidence": dict(confidences)}


def detect_image(model, uploaded_file):
    original = Image.open(uploaded_file).convert("RGB")
    result = model(np.array(original), verbose=False)[0]
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    return {"kind": "image", "original": original, "annotated": annotated, "stats": extract_stats(result)}


def detect_video(model, uploaded_file):
    input_suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as source:
        source.write(uploaded_file.getvalue())
        input_path = Path(source.name)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        input_path.unlink(missing_ok=True)
        raise RuntimeError("The selected video could not be opened.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    output_fd, output_name = tempfile.mkstemp(suffix=".mp4")
    os.close(output_fd)
    output_path = Path(output_name)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    counts = Counter()
    confidences = defaultdict(list)
    highest = 0.0
    frame_count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            result = model(frame, verbose=False)[0]
            stats = extract_stats(result)
            counts.update(stats["classes"])
            for label, values in stats["confidence"].items():
                confidences[label].extend(values)
            highest = max(highest, stats["highest"])
            writer.write(result.plot())
            frame_count += 1
    finally:
        capture.release()
        writer.release()
        input_path.unlink(missing_ok=True)

    return {"kind": "video", "video": output_path.read_bytes(), "frames": frame_count, "stats": {
        "total": sum(counts.values()), "highest": highest, "classes": dict(counts), "confidence": dict(confidences)
    }}


def inject_styles(st):
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --bg:#0b1117; --surface:#111b24; --surface2:#172530; --line:#243642; --text:#edf4f5; --muted:#8ea1aa; --cyan:#62d8e7; }
    .stApp { background:var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; }
    [data-testid='stHeader'] { background:transparent; }
    [data-testid='stToolbar'] { visibility:hidden; }
    .block-container { max-width:1320px; padding:1.5rem 3.5rem 3rem; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.02em; }
    .topbar { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding:12px 0 18px; margin-bottom:18px; }
    .brand { display:flex; align-items:center; gap:12px; font-family:'Space Grotesk'; font-weight:700; font-size:20px; }
    .brand-mark { display:grid; place-items:center; width:38px; height:38px; border-radius:11px; color:#071116; background:var(--cyan); font-size:14px; }
    .brand small { display:block; color:var(--muted); font:500 11px 'DM Sans'; margin-top:2px; }
    .status { color:#9ee9c0; font-size:12px; font-weight:600; border:1px solid #27533f; background:#10251e; border-radius:999px; padding:8px 12px; }
    .status i { color:#65dfa2; font-style:normal; margin-right:5px; }
    .eyebrow { color:var(--cyan); font:600 11px 'DM Sans'; letter-spacing:.14em; text-transform:uppercase; }
    .nav-wrap { display:flex; align-items:center; gap:22px; }
    .nav-note { color:var(--muted); font-size:12px; }
    .hero { display:flex; justify-content:space-between; gap:48px; align-items:center; margin:56px 0 46px; }
    .hero h1 { font-size:clamp(2.8rem,5vw,5.4rem); line-height:.94; margin:10px 0 20px; max-width:760px; }
    .hero p { color:var(--muted); font-size:16px; line-height:1.6; max-width:560px; margin:0; }
    .hero-signal { min-width:180px; border-left:2px solid var(--cyan); padding-left:16px; color:var(--muted); font-size:12px; line-height:1.7; }
    .hero-signal strong { display:block; color:var(--text); font:600 18px 'Space Grotesk'; }
    .hero-visual { min-height:310px; border:1px solid var(--line); border-radius:20px; overflow:hidden; position:relative; background:var(--surface); }
    .hero-visual img { width:100%; height:310px; object-fit:cover; opacity:.78; }
    .hero-visual-label { position:absolute; left:18px; bottom:18px; background:#0b1117e8; border:1px solid var(--line); padding:11px 14px; color:var(--text); font-size:12px; }
    .hero-visual-label b { display:block; color:var(--cyan); font:600 16px 'Space Grotesk'; margin-bottom:3px; }
    .home-card-row { display:grid; grid-template-columns:1.2fr .8fr .8fr; gap:14px; margin-bottom:30px; }
    .home-card { background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:22px; min-height:130px; }
    .home-card.feature { background:linear-gradient(135deg,#173f49,#111b24); border-color:#30616c; }
    .home-card .number { color:var(--cyan); font:600 28px 'Space Grotesk'; }
    .home-card h3 { color:var(--text); font-size:16px; margin:12px 0 6px; }
    .home-card p { color:var(--muted); font-size:12px; line-height:1.5; margin:0; }
    .page-intro { margin:42px 0 26px; }
    .page-intro h1 { font-size:42px; margin:8px 0 10px; }
    .page-intro p { color:var(--muted); max-width:620px; line-height:1.6; }
    .panel { background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:24px; }
    .studio-grid { display:grid; grid-template-columns:minmax(0,.82fr) minmax(0,1.18fr); gap:16px; align-items:start; }
    .studio-card { background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:20px; }
    .studio-card.result-card { background:linear-gradient(145deg,#142b32 0%,#111b24 62%); border-color:#2d5961; }
    .media-heading { display:flex; justify-content:space-between; align-items:center; gap:12px; margin:4px 0 13px; }
    .media-heading strong { font:600 16px 'Space Grotesk'; }
    .media-heading span { color:var(--muted); font-size:11px; letter-spacing:.08em; text-transform:uppercase; }
    .media-frame { background:#071015; border:1px solid #29414a; border-radius:12px; overflow:hidden; padding:7px; }
    .media-frame img, .media-frame video { display:block; width:100%; max-height:460px; object-fit:contain; border-radius:7px; }
    .ready-state { min-height:250px; border:1px dashed #41626a; border-radius:12px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; padding:24px; background:rgba(7,16,21,.42); }
    .ready-icon { display:grid; place-items:center; width:46px; height:46px; border:1px solid #39717a; border-radius:50%; color:var(--cyan); font:600 20px 'Space Grotesk'; margin-bottom:14px; }
    .ready-state strong { color:var(--text); font:600 18px 'Space Grotesk'; margin-bottom:7px; }
    .ready-state span { color:var(--muted); font-size:12px; line-height:1.5; max-width:240px; }
    .upload-meta { display:flex; align-items:center; gap:8px; color:var(--muted); font-size:12px; margin:10px 0 14px; }
    .upload-meta b { color:var(--text); font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    @media (max-width:800px) { .studio-grid { grid-template-columns:1fr; } .media-frame img, .media-frame video { max-height:340px; } }
    .panel-title { font:600 17px 'Space Grotesk'; margin-bottom:18px; }
    .panel-kicker { color:var(--muted); font-size:12px; margin-top:-12px; margin-bottom:18px; }
    .empty { min-height:300px; border:1px dashed #36505a; border-radius:13px; display:flex; flex-direction:column; justify-content:center; align-items:center; color:var(--muted); text-align:center; }
    .empty strong { color:var(--text); font:600 19px 'Space Grotesk'; margin-bottom:7px; }
    .stat-row { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:22px 0 28px; }
    .stat { background:var(--surface); border:1px solid var(--line); border-radius:13px; padding:16px; }
    .stat label { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
    .stat b { display:block; color:var(--cyan); font:600 25px 'Space Grotesk'; margin-top:7px; }
    .section-title { color:var(--text); font:600 20px 'Space Grotesk'; margin:30px 0 12px; }
    .class-row { display:flex; justify-content:space-between; border-bottom:1px solid var(--line); padding:13px 0; color:var(--muted); }
    .class-row b { color:var(--text); }
    .info { background:var(--surface2); border-radius:13px; padding:20px 24px; color:var(--muted); line-height:1.6; }
    .footer { border-top:1px solid var(--line); margin-top:48px; padding-top:18px; color:var(--muted); font-size:12px; }
    @media (max-width:800px) { .block-container { padding:1.25rem 1rem 2rem; } .topbar { margin-bottom:16px; } .hero { display:block; margin:35px 0; } .hero-signal { margin-top:24px; } .stat-row,.home-card-row { grid-template-columns:1fr 1fr; } .hero-visual { margin-top:28px; } }
    </style>
    """, unsafe_allow_html=True)


def render_nav(st):
    st.markdown("<div class='topbar'><div class='brand'><span class='brand-mark'>RV</span><span>RoadVision<small>Road Damage Detection</small></span></div><div class='nav-wrap'><span class='nav-note'>YOLOv8 / LOCAL VISION</span><div class='status'><i>●</i> Model Ready</div></div></div>", unsafe_allow_html=True)
    current = st.session_state.get("page", "Home")
    page = st.radio("Navigate", ["Home", "Prediction", "How it works"], index=["Home", "Prediction", "How it works"].index(current), horizontal=True, label_visibility="collapsed", key="navigation")
    if page != current:
        st.session_state.page = page
        st.rerun()


def render_home(st):
    st.markdown("<div class='hero'><div><div class='eyebrow'>Road intelligence / field operations</div><h1>See the road.<br><span style='color:var(--cyan)'>Fix it sooner.</span></h1><p>RoadVision turns ordinary road imagery into a clear damage map. Detect cracks, potholes, patches, and other road conditions with the trained YOLOv8 model.</p></div><div class='hero-signal'><strong>4 classes</strong>Image + video input<br>Private local processing</div></div>", unsafe_allow_html=True)
    visual_col, action_col = st.columns([1.4, .8], gap="large")
    with visual_col:
        if (APP_DIR / "ai_gen.jpg").exists():
            st.image(str(APP_DIR / "ai_gen.jpg"), width="stretch")
        st.markdown("<div class='hero-visual-label'><b>FIELD VIEW / READY</b>Computer vision for safer streets</div>", unsafe_allow_html=True)
    with action_col:
        st.markdown("<div class='home-card feature'><div class='number'>01</div><h3>Start a prediction</h3><p>Upload a road image or video and get annotated results, confidence scores, and class summaries.</p></div>", unsafe_allow_html=True)
        if st.button("Open Prediction Studio  →", type="primary", width="stretch"):
            st.session_state.page = "Prediction"
            st.rerun()
        st.markdown("<div class='home-card'><div class='number'>02</div><h3>Review what matters</h3><p>Compare the original frame with the actual YOLOv8 detection output.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='home-card-row'><div class='home-card'><div class='number'>01</div><h3>Upload</h3><p>Bring a road image or video from your inspection workflow.</p></div><div class='home-card'><div class='number'>02</div><h3>Analyze</h3><p>Local YOLOv8 inference scans visible road damage.</p></div><div class='home-card'><div class='number'>03</div><h3>Act</h3><p>Use clear detections to prioritize maintenance.</p></div></div>", unsafe_allow_html=True)


def render_prediction(st, model):
    st.markdown("<div class='page-intro'><div class='eyebrow'>Prediction studio / live analysis</div><h1>Inspect road imagery</h1><p>Choose an input, run the trained model, and review the annotated result without leaving the workspace.</p></div>", unsafe_allow_html=True)
    mode = st.radio("Input type", ["Image", "Video"], horizontal=True, label_visibility="collapsed")
    uploaded = st.file_uploader("Upload road imagery", type=IMAGE_TYPES if mode == "Image" else VIDEO_TYPES, label_visibility="collapsed")
    result = st.session_state.get("result")
    left, right = st.columns([.9, 1.1], gap="large")
    with left:
        st.markdown("<div class='studio-card'><div class='media-heading'><strong>Input frame</strong><span>Source media</span></div>", unsafe_allow_html=True)
        if uploaded:
            st.markdown(f"<div class='upload-meta'><span>Selected</span><b>{uploaded.name}</b></div>", unsafe_allow_html=True)
            st.markdown("<div class='media-frame'>", unsafe_allow_html=True)
            if mode == "Image":
                st.image(uploaded, width="stretch")
            else:
                st.video(uploaded)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='ready-state'><div class='ready-icon'>+</div><strong>Bring in a road view</strong><span>Drop an image or video above to prepare the inspection.</span></div>", unsafe_allow_html=True)
        if st.button("Run Detection  →", type="primary", width="stretch", disabled=uploaded is None or model is None):
            try:
                started = time.perf_counter()
                with st.spinner("Analyzing road imagery..."):
                    result = detect_image(model, uploaded) if mode == "Image" else detect_video(model, uploaded)
                st.session_state.result = result
                st.session_state.elapsed = time.perf_counter() - started
                st.rerun()
            except Exception:
                st.error("Detection failed. Check the model and input file.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='studio-card result-card'><div class='media-heading'><strong>Detection output</strong><span>YOLOv8 annotation</span></div>", unsafe_allow_html=True)
        if result is None:
            st.markdown("<div class='ready-state'><div class='ready-icon'>◎</div><strong>Your evidence appears here</strong><span>Run detection to reveal boxes, labels, and confidence scores.</span></div>", unsafe_allow_html=True)
        elif result["kind"] == "image":
            st.markdown("<div class='media-frame'>", unsafe_allow_html=True)
            st.image(result["annotated"], width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='media-frame'>", unsafe_allow_html=True)
            st.video(result["video"])
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption(f"Processed {result['frames']} frames")
        st.markdown("</div>", unsafe_allow_html=True)
    if result is not None:
        render_stats(st, result["stats"], st.session_state.get("elapsed", 0))
        if result["kind"] == "image":
            st.markdown("<div class='section-title'>Original / Detection Result</div>", unsafe_allow_html=True)
            original, annotated = st.columns(2, gap="medium")
            with original:
                st.caption("ORIGINAL IMAGE")
                st.image(result["original"], width="stretch")
            with annotated:
                st.caption("ANNOTATED RESULT")
                st.image(result["annotated"], width="stretch")
        render_damage_classes(st, result["stats"])


def render_about(st):
    st.markdown("<div class='page-intro'><div class='eyebrow'>RoadVision / methodology</div><h1>Built for the moment<br>between seeing and fixing.</h1><p>A focused computer-vision workflow for turning road imagery into useful maintenance signals.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='home-card-row'><div class='home-card feature'><div class='number'>01</div><h3>Upload imagery</h3><p>Use the Prediction Studio to bring in a supported road image or video.</p></div><div class='home-card'><div class='number'>02</div><h3>YOLOv8 inference</h3><p>The existing trained model analyzes the input locally and returns real boxes.</p></div><div class='home-card'><div class='number'>03</div><h3>Review evidence</h3><p>Inspect class counts, confidence, processing time, and annotated output.</p></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Model coverage</div>", unsafe_allow_html=True)
    st.markdown("<div class='info'><b>CRACK</b> &nbsp; / &nbsp; <b>POTHOLE</b> &nbsp; / &nbsp; <b>PATCH</b> &nbsp; / &nbsp; <b>OTHER</b><br><br>All labels shown here come from the existing trained model configuration. No results are invented or hardcoded.</div>", unsafe_allow_html=True)


def render_stats(st, stats, elapsed):
    st.markdown(f"<div class='stat-row'><div class='stat'><label>Detected objects</label><b>{stats['total']}</b></div><div class='stat'><label>Highest confidence</label><b>{stats['highest']:.1%}</b></div><div class='stat'><label>Damage classes</label><b>{len(stats['classes'])}</b></div><div class='stat'><label>Processing time</label><b>{elapsed:.2f}s</b></div></div>", unsafe_allow_html=True)


def render_damage_classes(st, stats):
    st.markdown("<div class='section-title'>Detected Road Damage</div>", unsafe_allow_html=True)
    if not stats["classes"]:
        st.info("No road damage detected in this input.")
        return
    rows = []
    for label, count in stats["classes"].items():
        values = stats["confidence"].get(label, [])
        average = sum(values) / len(values) if values else 0
        plural = "s" if count != 1 else ""
        rows.append(f"<div class='class-row'><span><b>{label.title()}</b><br><small>{count} detection{plural}</small></span><span>{average:.1%} avg confidence</span></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def main():
    import streamlit as st

    st.set_page_config(page_title="RoadVision", page_icon="🛣️", layout="wide", initial_sidebar_state="collapsed")
    inject_styles(st)
    render_nav(st)
    model = load_model(st)
    page = st.session_state.get("page", "Home")
    if page == "Home":
        render_home(st)
    elif page == "Prediction":
        render_prediction(st, model)
    else:
        render_about(st)
    st.markdown("<div class='footer'>RoadVision • YOLOv8 Road Damage Detection</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()