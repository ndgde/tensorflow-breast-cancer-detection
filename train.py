import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import plot_model
import os
import pandas as pd
import pydicom
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Настройки
NUM_CLASSES = 6  # BI-RADS классы
BATCH_SIZE = 64
EPOCHS = 50
RESIZE_VALUE = 224 # изображение приводится к этому размеру
DICOM_DIR = "INbreast Release 1.0/ALLDICOMs"
CSV_FILE = "INbreast Release 1.0/INbreast.csv"
VALID_INDICES_FILE = "valid_patient_indices.csv"

# Функция для загрузки данных из DICOM и CSV
def load_data(dicom_dir, csv_file, valid_indices_file):
    # Загрузка валидных индексов пациентов
    valid_indices = pd.read_csv(valid_indices_file)['Patient Index'].astype(str).tolist()

    # Загрузка CSV с дополнительной информацией
    metadata = pd.read_csv(csv_file, delimiter=';')
    metadata = metadata.rename(columns=lambda x: x.strip())

    # Очистка и приведение данных
    metadata['File Name'] = pd.to_numeric(metadata['File Name'], errors='coerce')
    metadata['ACR'] = pd.to_numeric(metadata['ACR'], errors='coerce')
    metadata['Bi-Rads'] = pd.to_numeric(metadata['Bi-Rads'], errors='coerce')

    metadata = metadata.dropna(subset=['File Name', 'ACR', 'Bi-Rads'])
    metadata['File Name'] = metadata['File Name'].astype(int)
    metadata['ACR'] = metadata['ACR'].astype(int)
    metadata['Bi-Rads'] = metadata['Bi-Rads'].astype(int)

    data = []
    labels = []

    patient_files = os.listdir(dicom_dir)
    for patient_index in tqdm(valid_indices, desc="Loading data"):
        patient_files_subset = [f for f in patient_files if f"_{patient_index}_" in f]

        for file in patient_files_subset:
            dicom_path = os.path.join(dicom_dir, file)
            ds = pydicom.dcmread(dicom_path)

            # Извлечение изображения и информации
            pixel_array = ds.pixel_array
            img_res = np.array(pixel_array.shape)  # Разрешение изображения
            left_right = 0 if "_L_" in file else 1  # 0 - левая, 1 - правая

            # Извлечение данных из CSV
            file_name = int(file.split('_')[0])
            row = metadata[metadata['File Name'] == file_name]

            if not row.empty:
                breast_density = row.iloc[0]['ACR']
                bi_rads = row.iloc[0]['Bi-Rads']

                # Подготовка данных
                data.append((pixel_array, left_right, breast_density, img_res))
                labels.append(bi_rads - 1)  # BI-RADS 1-6 -> 0-5

    return data, labels

# Подготовка данных для TensorFlow
def preprocess_data(data, labels):
    images = []
    left_right = []
    breast_density = []
    img_res = []

    for (img, lr, bd, res) in tqdm(data, desc="Preprocessing data"):
        # Преобразование изображения в нужный размер
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = tf.expand_dims(img, axis=-1)  # Добавляем канал
        img = tf.image.resize(img, (RESIZE_VALUE, RESIZE_VALUE))  # Resize изображений
        img = tf.image.grayscale_to_rgb(img)  # Конвертируем в RGB

        images.append(img)
        left_right.append([lr])
        breast_density.append([bd])
        img_res.append(res)

    return (
        [
            tf.convert_to_tensor(left_right, dtype=tf.float32),
            tf.convert_to_tensor(breast_density, dtype=tf.float32),
            tf.convert_to_tensor(img_res, dtype=tf.float32),
            tf.convert_to_tensor(images, dtype=tf.float32),
            tf.convert_to_tensor(images, dtype=tf.float32),  # img1 и img2 в данном примере одинаковые
        ],
        tf.convert_to_tensor(labels, dtype=tf.int32),
    )

# Создание модели с предобученной базой
def create_model():
    base_model = EfficientNetB0(include_top=False, input_shape=(RESIZE_VALUE, RESIZE_VALUE, 3), weights='imagenet')
    base_model.trainable = False

    img_input = layers.Input(shape=(RESIZE_VALUE, RESIZE_VALUE, 3), name="img")
    x = base_model(img_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Дополнительные данные
    left_right_input = layers.Input(shape=(1,), name='left_right')
    breast_density_input = layers.Input(shape=(1,), name='breast_density')
    img_res_input = layers.Input(shape=(2,), name='img_resolution')

    combined = layers.concatenate([x, left_right_input, breast_density_input, img_res_input])

    # Полносвязные слои
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs=[left_right_input, breast_density_input, img_res_input, img_input, img_input], outputs=output)
    return model

# Загрузка данных
print("Loading data...")
data, labels = load_data(DICOM_DIR, CSV_FILE, VALID_INDICES_FILE)
print("Data loaded.")

print("Preprocessing data...")
inputs, labels = preprocess_data(data, labels)
print("Data preprocessing complete.")

# Разделение на обучение и тест
train_size = int(0.8 * len(labels))
train_inputs = [tf.convert_to_tensor(inp[:train_size]) for inp in inputs]
test_inputs = [tf.convert_to_tensor(inp[train_size:]) for inp in inputs]
train_labels = labels[:train_size]
test_labels = labels[train_size:]

# Создание и компиляция модели
print("Creating model...")
model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Обучение модели
print("Training model...")
history = model.fit(train_inputs, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(test_inputs, test_labels))

# Сохранение модели
print("Saving model...")
model.save("breast_cancer_model.h5")

# Оценка модели
print("Evaluating model...")
loss, accuracy = model.evaluate(test_inputs, test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Визуализация метрик обучения
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Потеря
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Точность
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Сохранение графиков
plot_training_history(history)
# plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)