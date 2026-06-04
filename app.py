import streamlit as st
from modules.text_module import TextGSTPredictor
from modules.image_module import ImageGSTPredictor
from modules.ocr_module import OCRGSTVerifier
import os

DATASET_PATH = "data/GST_2025_HSN_SAC_Master_Cleaned.xlsx"

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart GST Chatbot",
    page_icon="🧾",
    layout="wide"
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .chat-bubble-user {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: white;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 6px 0;
        max-width: 75%;
        margin-left: auto;
        text-align: right;
    }
    .chat-bubble-bot {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        color: #e2e8f0;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 16px;
        margin: 6px 0;
        max-width: 85%;
        border-left: 3px solid #10b981;
    }
    .result-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 4px solid #10b981;
    }
    .result-title { font-size: 18px; font-weight: bold; color: #10b981; }
    .result-row   { font-size: 14px; color: #cbd5e1; margin-top: 6px; }
    .gst-badge {
        display: inline-block;
        background: #7c3aed;
        color: white;
        padding: 2px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .confidence-bar { margin-top: 6px; color: #94a3b8; font-size: 12px; }
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .ocr-field {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        display: flex;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧾 GST Chatbot")
    st.markdown("*Smart HSN Code Recommender*")
    st.markdown("---")

    module = st.radio(
        "📌 Choose Module",
        ["💬 Text Query", "🖼️ Image Upload", "📄 Invoice OCR", "✅ GSTIN Validator"],
        index=0
    )

    st.markdown("---")
    st.markdown("**📊 Dataset**")
    st.markdown("21,791 HSN/SAC codes")
    st.markdown("GST 2025 Master Data")
    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with Streamlit · TF-IDF · OCR")

# ─────────────────────────────────────────────
#  Session State for Chat History
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────
#  Helper — render result cards
# ─────────────────────────────────────────────
def render_result_card(match: dict, rank: int = 1):
    confidence = match.get("confidence", 0)
    bar = "🟩" * int(confidence // 20) + "⬜" * (5 - int(confidence // 20))
    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">#{rank} &nbsp; HSN: {match['hsn']} &nbsp;
            <span class="gst-badge">GST {match['gst_rate']}</span>
        </div>
        <div class="result-row">📝 {match['description']}</div>
        <div class="confidence-bar">{bar} Confidence: {confidence}%</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODULE 1 — Text Query (Chatbot)
# ─────────────────────────────────────────────
if "💬 Text Query" in module:
    st.markdown('<div class="section-header">💬 GST Chatbot — Text Query</div>', unsafe_allow_html=True)
    st.markdown("Type a product name or description to get the HSN code and GST rate.")

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">🧑 {chat["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-bot">🤖 {chat["text"]}</div>', unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("", placeholder="e.g. mobile phone, cotton shirt, laptop, rice...")
        with col2:
            submitted = st.form_submit_button("Send 🚀", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "text": user_input})

        with st.spinner("🔍 Searching HSN database..."):
            predictor = TextGSTPredictor(DATASET_PATH)
            results = predictor.predict(user_input, top_n=3)

        if results:
            best = results[0]
            bot_reply = (
                f"Found **{len(results)} matches** for *'{user_input}'*. "
                f"Best match → HSN **{best['hsn']}** | GST Rate: **{best['gst_rate']}** "
                f"| Confidence: **{best['confidence']}%**"
            )
            st.session_state.chat_history.append({"role": "bot", "text": bot_reply})

            st.markdown("### 🏆 Top Matches")
            for i, match in enumerate(results, 1):
                render_result_card(match, i)
        else:
            st.session_state.chat_history.append({
                "role": "bot",
                "text": "Sorry, no HSN code found for that description. Try a different keyword."
            })
        st.rerun()


# ─────────────────────────────────────────────
#  MODULE 2 — Image Upload
# ─────────────────────────────────────────────
elif "🖼️ Image Upload" in module:
    st.markdown('<div class="section-header">🖼️ Image-Based HSN Predictor</div>', unsafe_allow_html=True)
    st.markdown("Upload a product image — AI will describe it and find the HSN code.")

    uploaded_img = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        temp_path = "temp_product.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_img.read())

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(temp_path, caption="Uploaded Product", use_container_width=True)

        with col2:
            with st.spinner("🤖 Analyzing image and finding HSN code..."):
                predictor = ImageGSTPredictor(DATASET_PATH)
                result = predictor.predict(temp_path)

            detected_desc = result.get("detected_description", "")
            st.info(f"🔍 **AI detected:** *{detected_desc}*")

            st.markdown("### 🏆 Top HSN Matches")
            for i, match in enumerate(result.get("matches", []), 1):
                render_result_card(match, i)

        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ─────────────────────────────────────────────
#  MODULE 3 — Invoice OCR
# ─────────────────────────────────────────────
elif "📄 Invoice OCR" in module:
    st.markdown('<div class="section-header">📄 Invoice OCR Scanner</div>', unsafe_allow_html=True)
    st.markdown("Upload a GST invoice image to extract all tax details automatically.")

    # Tesseract status check
    import shutil, os
    tesseract_found = (
        shutil.which("tesseract") is not None or
        os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe") or
        os.path.exists(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe")
    )
    if not tesseract_found:
        st.warning(
            "⚠️ **Tesseract not detected!** OCR may fail. "
            "Install EasyOCR as backup: `pip install easyocr` "
            "OR install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki"
        )
    else:
        st.success("✅ Tesseract OCR detected and ready!")

    uploaded_invoice = st.file_uploader("Upload Invoice Image", type=["jpg", "jpeg", "png"])

    if uploaded_invoice:
        temp_invoice = "temp_invoice.jpg"
        with open(temp_invoice, "wb") as f:
            f.write(uploaded_invoice.read())

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(temp_invoice, caption="Uploaded Invoice", use_container_width=True)

        with col2:
            with st.spinner("📖 Reading invoice with OCR..."):
                verifier = OCRGSTVerifier()
                raw_text = verifier.extract_text(temp_invoice)
                info = verifier.extract_gst_info(raw_text)

            # Show raw text immediately so user can verify OCR worked
            with st.expander("🔍 View Raw OCR Text (check here if fields show Not Found)", expanded=True):
                if len(raw_text.strip()) > 20:
                    st.code(raw_text, language=None)
                else:
                    st.error("❌ OCR extracted no text! Tesseract may not be installed correctly. Try: pip install easyocr")

            st.markdown("### 📋 Extracted Invoice Details")

            icons = {
                "GSTIN Numbers Found": "🏢",
                "Invoice Number":      "🔢",
                "Invoice Date":        "📅",
                "Taxable Amount":      "💰",
                "CGST":                "📊",
                "SGST":                "📊",
                "IGST":                "📊",
                "Total Amount":        "💵",
                "HSN Codes in Invoice":"🔖",
                "Place of Supply":     "📍",
            }

            for field, value in info.items():
                icon = icons.get(field, "📌")
                display_val = ", ".join(value) if isinstance(value, list) else value
                st.markdown(f"""
                <div class="ocr-field">
                    <span style="color:#94a3b8">{icon} {field}</span>
                    <span style="color:#10b981; font-weight:bold">{display_val}</span>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📃 View Raw OCR Text"):
                st.text_area("Raw Text", raw_text, height=200)

        if os.path.exists(temp_invoice):
            os.remove(temp_invoice)


# ─────────────────────────────────────────────
#  MODULE 4 — GSTIN Validator
# ─────────────────────────────────────────────
elif "✅ GSTIN Validator" in module:
    st.markdown('<div class="section-header">✅ GSTIN Validator</div>', unsafe_allow_html=True)
    st.markdown("Enter a GSTIN number to validate and decode its details.")

    gstin_input = st.text_input("Enter GSTIN Number", placeholder="e.g. 33AAAAA0000A1Z5", max_chars=15)

    if st.button("Validate GSTIN 🔍", type="primary"):
        if len(gstin_input.strip()) != 15:
            st.error("❌ GSTIN must be exactly 15 characters!")
        else:
            verifier = OCRGSTVerifier()
            result = verifier.validate_gstin(gstin_input)

            col1, col2 = st.columns(2)
            col1.metric("GSTIN", result["GSTIN"])
            col2.metric("Status", result["Valid"])

            st.markdown(f"""
            <div class="result-card">
                <div class="result-row">🏛️ <b>State:</b> {result['State']}</div>
                <div class="result-row">🪪 <b>PAN Number:</b> {result['PAN']}</div>
                <div class="result-row">🔢 <b>State Code:</b> {result['GSTIN'][:2]}</div>
                <div class="result-row">🏢 <b>Entity Type:</b> {result['GSTIN'][12]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📖 GSTIN Structure Guide")
            st.markdown("""
            | Position | Characters | Meaning |
            |---|---|---|
            | 1–2 | Digits | State Code |
            | 3–12 | Alphanumeric | PAN Number |
            | 13 | Digit | Entity Number |
            | 14 | Z | Default 'Z' |
            | 15 | Alphanumeric | Check Digit |
            """)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("🧾 Smart GST Chatbot | HSN Code Recommender | GST 2025 | Built with Python + Streamlit")
