import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import os
from pdf2image import convert_from_path
import tempfile
import cv2
import numpy as np

# Set Tesseract path (Adjust for your environment)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Streamlit app setup
st.set_page_config(page_title="TableSnap: Enhanced OCR to CSV", layout="centered")
st.title("📸 TableSnap: Enhanced OCR to CSV")
st.markdown("""
<style>
.neon-heading {
  font-family: 'Poppins', sans-serif;
  font-size: 3rem;
  color: #00D1FF;
  text-align: center;
  text-shadow: 0 0 5px #00D1FF, 0 0 10px #00D1FF, 0 0 20px #9B4DFF, 0 0 30px #9B4DFF;
}
.stButton>button, .stDownloadButton>button {
    background-color: #00D1FF;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    transition: all 0.3s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background-color: #9B4DFF;
    transform: scale(1.05);
    box-shadow: 0 4px 15px rgba(155, 77, 255, 0.4);
    color: white;
}
</style>
<h1 class="neon-heading">Welcome to TableSnap!</h1>
""", unsafe_allow_html=True)

# Sidebar settings
st.sidebar.header("⚙️ Settings")
preprocess = st.sidebar.checkbox("🧪 Enable Preprocessing", value=True)
ocr_lang = st.sidebar.selectbox("🌐 OCR Language", options=["eng", "hin", "ori"], 
                                format_func=lambda x: {"eng": "English", "hin": "Hindi", "ori": "Odia"}[x])

uploaded_file = st.file_uploader("📤 Upload scanned image or PDF", type=["jpg", "jpeg", "png", "pdf"])

# Preprocessing function
def preprocess_image(image_pil):
    """Preprocess image to improve OCR accuracy."""
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Deskew the image
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Thresholding
    blur = cv2.GaussianBlur(deskewed, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

def extract_table_from_text(text):
    """Extract table-like data from OCR text."""
    lines = text.strip().split("\n")
    table_data = []
    for line in lines:
        # Split lines by consistent spaces or tabs
        row = [cell.strip() for cell in line.split() if cell.strip()]
        if len(row) > 1:  # Assume rows with more than one cell are table rows
            table_data.append(row)
    return table_data

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
    images = []

    # Handle PDF files
    if file_ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(uploaded_file.read())
            tmp_pdf_path = tmp_pdf.name
        images = convert_from_path(tmp_pdf_path, dpi=300)
    else:
        images.append(Image.open(uploaded_file))

    st.success(f"✅ {len(images)} page(s) ready for processing!")

    all_data = []

    for idx, img in enumerate(images):
        st.image(img, caption=f"Page {idx + 1}", use_container_width=True)

        if preprocess:
            img = preprocess_image(img)

        # Perform OCR
        text = pytesseract.image_to_string(img, lang=ocr_lang, config="--psm 6")
        
        # Debug: Display raw OCR output
        st.text_area(f"🔍 Raw OCR Output: Page {idx + 1}", text)

        # Extract table data
        table_data = extract_table_from_text(text)
        all_data.extend(table_data)

    # Convert extracted data to DataFrame
    if all_data:
        # Filter rows with more than one element
        table_rows = [row for row in all_data if len(row) > 1]
        
        if table_rows:
            max_columns = max(len(row) for row in table_rows)
            data_padded = [row + [''] * (max_columns - len(row)) for row in table_rows]
            df = pd.DataFrame(data_padded)

            # Clean column headers
            df.columns = [f"Column_{i + 1}" for i in range(len(df.columns))]

            st.subheader("📊 Extracted Table")
            st.dataframe(df)

            # Provide download option
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download as CSV", csv_data, "tablesnap_output.csv", "text/csv", key='download-csv')

            if st.button("📄 Process Another File"):
                st.experimental_rerun()
        else:
            st.warning("⚠️ No valid table rows found. Ensure the uploaded file contains a clear table structure.")
            st.stop()  # Stop if no valid rows
    else:
        st.warning("⚠️ No table-like structure found! Try enabling preprocessing or using a clearer scan.")
