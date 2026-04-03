#1. importing libraries
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import streamlit.components.v1 as components

#2. Page setup
st.set_page_config(page_title="✨ AI Photo Enhancer", page_icon="📸", layout="wide")

st.title("✨ AI Photo Enhancer")
st.write("Enhance your images with AI-powered filters 🚀")

#3. Google Analytics
components.html("""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-66VX2V57GZ"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'G-66VX2V57GZ');
</script>
""", height=0)

#4. Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

#5. Mode Selection
mode = st.selectbox(
    "Choose Enhancement Type",
    [
        "Soft (Light Improve)",
        "Normal (Balanced)",
        "Strong (High Enhance)",
        "Night Enhance (Low Light)",
        "High Detail (Sharp Focus)",
        "Warm Tone (Instagram Style)"
    ]
)

#6. If image uploaded
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    # Convert RGB → BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    if st.button("🚀 Enhance Image"):

        #7. Mode-based settings
        if "Soft" in mode:
            h, alpha, beta, clip = 5, 1.02, 3, 1.5
        elif "Normal" in mode:
            h, alpha, beta, clip = 7, 1.05, 5, 2.0
        elif "Strong" in mode:
            h, alpha, beta, clip = 10, 1.1, 10, 3.0
        elif "Night" in mode:
            h, alpha, beta, clip = 8, 1.08, 15, 3.5
        elif "High Detail" in mode:
            h, alpha, beta, clip = 6, 1.1, 5, 2.5
        else:
            h, alpha, beta, clip = 7, 1.05, 5, 2.0

        #8. Denoising
        denoised = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

        #9. Sharpening
        kernel = np.array([[0, -0.5, 0],
                           [-0.5, 3, -0.5],
                           [0, -0.5, 0]])

        sharp = cv2.filter2D(denoised, -1, kernel)
        sharpened = cv2.addWeighted(denoised, 0.8, sharp, 0.2, 0)

        #10. LAB color
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        #11. CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
        cl = clahe.apply(l)

        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        #12. Brightness + Contrast
        final = cv2.convertScaleAbs(enhanced_img, alpha=alpha, beta=beta)

        #13. Warm tone
        if "Warm" in mode:
            final[:,:,2] = cv2.add(final[:,:,2], 15)
            final[:,:,1] = cv2.add(final[:,:,1], 5)

        # Convert to RGB
        final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Enhanced Image")
            st.image(final_rgb, use_column_width=True)

        #14. Download
        result = Image.fromarray(final_rgb)
        buf = io.BytesIO()
        result.save(buf, format="PNG")

        st.download_button("📥 Download Image", buf.getvalue(), "enhanced.png")
