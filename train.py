import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import datetime
import cv2
import json
from sklearn.utils import class_weight

TF_ENABLE_ONEDNN_OPTS = 0
tf.config.run_functions_eagerly(True)

data_dir_input = input("Enter the full path to your training dataset folder: ")
data_dir = data_dir_input.strip()
img_size = 150
batch_size = 32
val_split = 0.15
epochs = 30
log_file = open("HAID_training_log.txt", "w", encoding="utf-8")

def log(text):
    print(text)
    log_file.write(f"{datetime.datetime.now()}: {text}\n")

train_aug = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    channel_shift_range=20.0,
    fill_mode='nearest',
    validation_split=val_split
)

train_gen = train_aug.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = train_aug.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(class_weights))

if os.path.exists("HAIDmodel.h5"):
    log("Loading existing model...")
    model = load_model("HAIDmodel.h5")
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
else:
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('HAIDmodel.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
]

log("Starting Training")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

val_gen.reset()
y_probs = model.predict(val_gen)
y_preds = (y_probs > 0.5).astype(int).flatten()
y_true = val_gen.classes

val_accuracy = np.mean(y_preds == y_true)
log(f"\nFinal Validation Accuracy: {val_accuracy*100:.2f}%")
log("\nClassification Report:\n")
log(classification_report(y_true, y_preds, target_names=['Normal', 'Cancer']))

history_all = {
    "accuracy": history.history['accuracy'],
    "val_accuracy": history.history['val_accuracy'],
    "loss": history.history['loss'],
    "val_loss": history.history['val_loss']
}

with open("training_history.json", "w") as f:
    json.dump(history_all, f)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_all['accuracy'], label='Train')
plt.plot(history_all['val_accuracy'], label='Validation')
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_all['loss'], label='Train')
plt.plot(history_all['val_loss'], label='Validation')
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()

def generate_heatmap(model, image_array, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([image_array]))
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    processed = tf.keras.applications.efficientnet.preprocess_input(img_array)
    heatmap = generate_heatmap(model, processed)
    heatmap = cv2.resize(heatmap, (img_size, img_size))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_array
    cv2.imwrite("heatmap_output.jpg", superimposed_img)
    log("Heatmap saved as heatmap_output.jpg")

log_file.close()
