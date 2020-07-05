import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance

edited_img = 0


def resize(img, height, width):
    img_new = cv2.resize(img, (height, width))
    return img_new


def denoise(img):
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return dst


def sharpen(img, val):
    # im = Image.open("car2.jpg")
    enhancer = ImageEnhance.Sharpness(img)
    enhanced_im = enhancer.enhance(val)
    return enhanced_im


def main():
    st.title("Gallery!!")
    uploaded_img = st.file_uploader("Choose an image...", type='jpg')

    if uploaded_img is not None:
        image = Image.open(uploaded_img)
        np_image = np.array(image)
        st.image(image, caption="Uploaded image", use_column_width=True)

        # --------------------------------------------------------------------------------------------- #

        width = np_image.shape[1]
        height = np_image.shape[0]
        width = int(st.text_input("width", width))
        height = int(st.text_input("height", height))
        if st.button("Resize"):
            resized_img = resize(np_image, width, height)
            st.image(resized_img, caption="Resized Image", use_column_width=False)

        # --------------------------------------------------------------------------------------------- #

        if st.button("Reduce Image Noise"):
            denoised_img = denoise(np_image)
            st.image([image, denoised_img], caption=["Original Image", "De-Noised Image"], use_column_width=True)

        # --------------------------------------------------------------------------------------------- #

        val = st.slider("Select intensity of sharpened image", 0.0, 10.0)
        if st.button("Sharpen Image"):
            sharpened_img = sharpen(image, val)
            st.image([image, sharpened_img], caption=["Original Image", "Sharpened Image"], use_column_width=True)


if __name__ == '__main__':
    main()
