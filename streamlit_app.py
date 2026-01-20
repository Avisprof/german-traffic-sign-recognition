import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
import os
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
BACKEND_URL = st.sidebar.text_input(
    "Backend URL",
    value=DEFAULT_BACKEND_URL,
    help="URL of the FastAPI backend service"
)

st.title("🚦 German Traffic Sign Recognition")
st.markdown("Upload an image of a traffic sign to get classification predictions")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a traffic sign image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image file containing a traffic sign"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    
    # Predict button
    if st.button("🔍 Classify Traffic Sign"):
        with st.spinner("Processing image..."):
            try:
                # Prepare file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Send request to backend
                response = requests.post(f"{BACKEND_URL}/predict", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    predictions = result["top_3"]
                    
                    # Display results
                    st.success("✅ Classification completed!")
                    
                    # Show top prediction prominently
                    top_pred = predictions[0]
                    st.markdown("### 🎯 Top Prediction")
                    st.markdown(f"**{top_pred['label_name']}** (Class {top_pred['label']})")
                    st.markdown(f"**Confidence:** {top_pred['proba']:.2%}")
                    
                    # Create horizontal bar chart
                    st.markdown("### 📊 Top 3 Predictions")
                    
                    # Extract data for visualization
                    labels = [f"{p['label']}: {p['label_name']}" for p in predictions]
                    probs = [p['proba'] * 100 for p in predictions]  # Convert to percentage
                    
                    # Create horizontal bar chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    y_pos = np.arange(len(labels))
                    bars = ax.barh(y_pos, probs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    
                    # Add percentage labels on bars
                    for i, (bar, prob) in enumerate(zip(bars, probs)):
                        ax.text(
                            prob + 0.5,  # x position (slightly right of bar)
                            i,  # y position
                            f'{prob:.2f}%',
                            va='center',
                            fontweight='bold'
                        )
                    
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(labels)
                    ax.set_xlabel('Probability (%)', fontsize=12)
                    ax.set_xlim(0, 105)  # Leave some space for labels
                    ax.set_title('Top 3 Traffic Sign Classifications', fontsize=14, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3, linestyle='--')
                    
                    # Invert y-axis to show highest probability at top
                    ax.invert_yaxis()
                    
                    st.pyplot(fig)
                    
                    # Display detailed results in a table
                    st.markdown("### 📋 Detailed Results")
                    st.table({
                        "Rank": [1, 2, 3],
                        "Class ID": [p['label'] for p in predictions],
                        "Class Name": [p['label_name'] for p in predictions],
                        "Probability": [f"{p['proba']:.4f}" for p in predictions],
                        "Confidence (%)": [f"{p['proba']:.2f}%" for p in predictions]
                    })
                    
                else:
                    st.error(f"❌ Error: Backend returned status code {response.status_code}")
                    st.text(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Could not connect to backend at {BACKEND_URL}")
                st.info("Make sure the FastAPI backend is running. You can start it with: `python app.py`")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

else:
    st.info("👆 Please upload an image file to get started")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>German Traffic Sign Recognition System</small>
    </div>
    """,
    unsafe_allow_html=True
)
