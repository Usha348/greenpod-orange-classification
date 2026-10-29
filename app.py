
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Orange Classification",
    page_icon="🍊"
)


st.title("🍊 Orange Classification System")

st.markdown(
    """
    ### AI-Powered Orange Image Classification

    Upload an orange image to identify its class.

    **Available classes:**
    - Blackspot
    - Canker
    - Fresh
    - Grenning
    """
)

st.divider()


uploaded_file = st.file_uploader(
    "Choose an orange image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Orange Image",
        use_container_width=True
    )


if uploaded_file is not None:
    if st.button("Predict Orange Class"):
        from PIL import Image
        import numpy as np
        import tensorflow as tf

        model = tf.keras.models.load_model(
            "orange_cnn_model.keras"
        )

        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((128, 128))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

        class_names = [
            "blackspot",
            "canker",
            "fresh",
            "grenning"
        ]

        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]

        confidence = float(prediction[0][predicted_index]) * 100

        st.success(f"Predicted Orange Class: {predicted_class}")
        st.write(f"Prediction Confidence: {confidence:.2f}%")
        st.progress(confidence / 100)

