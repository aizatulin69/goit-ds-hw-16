import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers #type: ignore
from keras.datasets import fashion_mnist
import random
from tensorflow.keras.applications import VGG16 #type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input #type: ignore
from PIL import Image as PILImage

CLASS_NAMES = ['Футболка/топ', 'Штани', 'Пуловер', 'Сукня', 'Пальто',
               'Сандалі', 'Сорочка', 'Кросівки', 'Сумка', 'Черевики']

def load_model_1():
    model = tf.keras.models.load_model('model1.keras')
    return model

def load_model_2():
    model = tf.keras.models.load_model('model2.keras')
    return model

def predict_model_1(model, image):
    image_normalized = image.astype(np.float32) / 255.0
    image_prepared = image_normalized.reshape(1, 28, 28, 1)
    predictions = model.predict(image_prepared, verbose=0)
    return predictions[0]

def predict_model_2(model, image):
    image_rgb = np.stack([image, image, image], axis=-1) 
    image_pil = PILImage.fromarray(image_rgb.astype('uint8'))
    image_resized = image_pil.resize((32, 32))
    image_array = np.array(image_resized)
    image_prepared = np.expand_dims(image_array, axis=0) 
    image_prepared = preprocess_input(image_prepared)
    predictions = model.predict(image_prepared, verbose=0)
    return predictions[0]

def main():
    st.set_page_config(
        page_title="Fashion MNIST Класифікатор",
        page_icon="👔",
        layout="wide"
    )
    
    st.title("Fashion MNIST Класифікатор")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Налаштування")
        
        model_choice = st.radio(
            "Оберіть архітектуру нейромережі:",
            ("Модель 1 (RNN)", "Модель 2 (VGG16)"),
            help="Оберіть одну з двох доступних архітектур"
        )
        
        st.markdown("---")
        st.markdown("### Про датасет")
        st.info("""
        **Fashion MNIST** містить 70,000 зображень одягу розміром 28x28 пікселів у відтінках сірого.
        
        **10 класів:**
        - Футболка/топ
        - Штани
        - Пуловер
        - Сукня
        - Пальто
        - Сандалі
        - Сорочка
        - Кросівки
        - Сумка
        - Черевики
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Завантаження зображення")
        
        upload_method = st.radio(
            "Оберіть спосіб завантаження:",
            ("Завантажити файл", "Використати зразок з датасету"),
            horizontal=True
        )
        
        image_array = None
        
        if upload_method == "Завантажити файл":
            uploaded_file = st.file_uploader(
                "Оберіть зображення (28x28, відтінки сірого)",
                type=['png', 'jpg', 'jpeg'],
                help="Завантажте зображення одягу розміром 28x28 пікселів"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('L')
                image = image.resize((28, 28))
                image_array = np.array(image)
                
        else:
            st.info("Для використання зразків встановіть: `pip install tensorflow`")
            
            try:
                import tensorflow as tf
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
                
                sample_idx = st.slider(
                    "Оберіть індекс зразка з тестового набору:",
                    0, len(x_test) - 1, 0
                )
                
                image_array = x_test[sample_idx]
                true_label = CLASS_NAMES[y_test[sample_idx]]
                st.success(f"Справжній клас: **{true_label}**")
                
            except ImportError:
                st.error("TensorFlow не встановлено. Встановіть його для завантаження зразків.")
        
        if image_array is not None:
            st.subheader("Завантажене зображення")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(image_array, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
    
    with col2:
        st.header("Результати класифікації")
        
        if image_array is not None:
            if st.button("Класифікувати", type="primary", use_container_width=True):
                with st.spinner("Обробка..."):
                    try:
                        if model_choice == "Модель 1":
                            model = load_model_1()
                            predictions = predict_model_1(model, image_array)
                        else:
                            model = load_model_2()
                            predictions = predict_model_2(model, image_array)
                        
                        if predictions is None:
                            st.error("Функція передбачення не реалізована. Додайте ваш код у МІСЦЕ 4 або МІСЦЕ 5.")
                        else:
                            predicted_class_idx = np.argmax(predictions)
                            predicted_class = CLASS_NAMES[predicted_class_idx]
                            confidence = predictions[predicted_class_idx] * 100
                            
                            st.success(f"### Передбачення: **{predicted_class}**")
                            st.metric("Впевненість", f"{confidence:.2f}%")
                            
                            st.markdown("---")
                            st.subheader("Імовірності по класах")
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            colors = ['#FF6B6B' if i == predicted_class_idx else '#4ECDC4' 
                                     for i in range(len(CLASS_NAMES))]
                            bars = ax.barh(CLASS_NAMES, predictions * 100, color=colors)
                            ax.set_xlabel('Імовірність (%)', fontsize=12)
                            ax.set_xlim(0, 100)
                            ax.grid(axis='x', alpha=0.3)
                            
                            for i, (bar, prob) in enumerate(zip(bars, predictions)):
                                if prob > 0.01:
                                    ax.text(prob * 100 + 1, i, f'{prob*100:.1f}%', 
                                           va='center', fontsize=9)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            with st.expander("Детальна таблиця імовірностей"):
                                prob_data = {
                                    "Клас": CLASS_NAMES,
                                    "Імовірність (%)": [f"{p*100:.2f}" for p in predictions]
                                }
                                st.table(prob_data)
                    
                    except Exception as e:
                        st.error(f"Помилка при класифікації: {str(e)}")
                        st.info("Помилка завантаження.")
        else:
            st.info("Завантажте зображення зліва для початку класифікації")
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Створено з використанням Streamlit | Fashion MNIST Dataset</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()