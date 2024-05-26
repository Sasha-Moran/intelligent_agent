import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image


class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# Завантаження збереженої моделі
loaded_model = load_model('cifar10_model.h5')

# Функція для завантаження зображення користувача та його передбачення
def predict_user_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img = np.array(img).astype("float32") / 255
    img = img.reshape(1, 32, 32, 3)

    prediction = loaded_model.predict(img)
    predicted_class = np.argmax(prediction)

    return class_names[predicted_class]

if __name__ == "__main__":
    # Виклик функції з шляхом до зображення
    image_path = 'dog.jpg'
    predicted_class = predict_user_image(image_path)
    print("Predicted class:", predicted_class)
