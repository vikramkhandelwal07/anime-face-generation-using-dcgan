import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

def load_generator(model_path):
    return tf.keras.models.load_model(model_path)

def generate_faces(generator, num_images, randomness_level):
    generated_images = []
    for _ in range(num_images):
        random_noise = np.random.normal(size=(1, 100)) * randomness_level
        generated_image = generator.predict(random_noise)[0]
        generated_image = (generated_image + 1) * 127.5
        generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(generated_image)
        generated_images.append(pil_image)
    return generated_images

def main():
    st.title("Anime Face Generator")
    generator = load_generator("generator_model.h5")

    num_images = st.slider("Number of Images", 1, 5, 2)
    randomness_level = st.slider("Randomness Level", 0.1, 1.2, 1.0, step=0.1)
    
    if st.button("Generate Faces"):
        generated_images = generate_faces(generator, num_images, randomness_level)
        st.write("Generated Anime Faces:")
        
        columns = st.columns(num_images)
        for i, (img, col) in enumerate(zip(generated_images, columns)):
            col.image(img, caption=f"Generated Face {i+1}", use_column_width=True)
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            col.download_button(label=f"Download {i+1}", data=img_bytes.getvalue(), file_name=f"anime_face_{i+1}.png", mime='image/png')

if __name__ == '__main__':
    main()
