import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import os
from pdf2image import convert_from_path
import tempfile

st.set_page_config(page_title="TableSnap: From Paper to Excel", layout="centered")
st.title("📸 TableSnap: From Paper to Excel in 10 Secs")

uploaded_file = st.file_uploader("Upload a scanned image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
    images = []

    # Handle PDFs
    if file_ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(uploaded_file.read())
            tmp_pdf_path = tmp_pdf.name
        images = convert_from_path(tmp_pdf_path, dpi=300)
    
    # Handle image files
    else:
        img = Image.open(uploaded_file)
        images.append(img)

    st.success(f"✅ {len(images)} page(s) ready for processing")

    all_data = []

    for idx, img in enumerate(images):
        st.image(img, caption=f"Page {idx+1}", use_container_width=True)

        text = pytesseract.image_to_string(img)
        lines = text.strip().split("\n")
        table_like = [line for line in lines if len(line.split()) > 1]

        for row in table_like:
            all_data.append(row.split())

    if all_data:
        max_len = max(len(row) for row in all_data)
        data_padded = [row + ['']*(max_len - len(row)) for row in all_data]

        df = pd.DataFrame(data_padded)
        st.subheader("📊 Extracted Table")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="tablesnap_output.csv",
            mime="text/csv",
            key="download_csv_button"  # 👈 FIXED: unique key
        )
    else:
        st.warning("⚠️ No table-like data found in the uploaded file.")
