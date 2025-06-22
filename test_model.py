import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report

model_path = "models/HAIDmodel.h5"
img_size = 150
test_dir = "test_samples"

model = load_model(model_path)

def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=(img_size, img_size))
            img_array = img_to_array(img)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
            images.append(img_array)
            labels.append(label)
    return images, labels

class0_imgs, class0_labels = load_images_from_folder(os.path.join(test_dir, "class0"), 0)
class1_imgs, class1_labels = load_images_from_folder(os.path.join(test_dir, "class1"), 1)

X_test = np.array(class0_imgs + class1_imgs)
y_true = np.array(class0_labels + class1_labels)

y_probs = model.predict(X_test)
y_preds = (y_probs > 0.5).astype(int).flatten()

print("\nClassification Report:\n")
print(classification_report(y_true, y_preds, target_names=["Normal", "Cancer"]))
