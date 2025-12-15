import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout,
                                     GlobalAveragePooling2D, Dense,
                                     LeakyReLU, BatchNormalization)

# ----------------------------------------------------------------
# CONFIGURACIÓN DEL SISTEMA
# ----------------------------------------------------------------
IMG_SIZE = 128
CHANNELS = 3
BATCH_SIZE = 64  # Reducido para CPU, aumentar a 256 cuando tengas GPU funcionando
EPOCHS = 35
LEARNING_RATE = 0.001

# Configurar TensorFlow para usar GPU eficientemente (comentado temporalmente)
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     print(f"GPU detectada: {tf.config.experimental.get_device_details(physical_devices[0])}")
# else:
#     print("No se detectó GPU, usando CPU")
print("Usando CPU para entrenamiento")

# Ruta al dataset
dataset_path = os.path.join(os.getcwd(), 'animals-dataset')

# ----------------------------------------------------------------
# PASO 1: Configuración de Generadores de Datos (Solución MemoryError)
# ----------------------------------------------------------------
print(f"Configurando generadores desde: {dataset_path}")

# Generador para entrenamiento con augmentación de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Generador para test (solo normalización, sin augmentación)
test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    interpolation='bilinear',      # Añadir método de interpolación
    keep_aspect_ratio=False        # Forzar redimensionamiento exacto, si es True mantiene proporciones y agrega borde negro
)

# Cargar datos de validación
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    interpolation='bilinear',
    keep_aspect_ratio=False
)

# Obtener nombres de clases y número de clases
class_names = list(train_generator.class_indices.keys())
nClasses = len(class_names)

print(f"\nClases detectadas: {class_names}")
print(f"Número total de clases: {nClasses}")
print(f"Imágenes de entrenamiento: {train_generator.samples}")
print(f"Imágenes de validación: {validation_generator.samples}")

# ----------------------------------------------------------------
# PASO 2: Definición de la Arquitectura CNN MEJORADA
# ----------------------------------------------------------------
model = Sequential()

# --- BLOQUE 1: Características de bajo nivel (Bordes, Colores) --- 128 -> 64
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# --- BLOQUE 2: Características medias (Texturas, Formas simples) --- 64 -> 32
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# --- BLOQUE 3: Características complejas (Partes de animales) --- 32 -> 16
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# --- BLOQUE 4: Características avanzadas (Animales completos) --- 16 -> 8
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# --- CLASIFICADOR (Top Model) ---
model.add(GlobalAveragePooling2D())  # Mejor que Flatten para evitar overfitting
model.add(Dense(512))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(nClasses, activation='softmax'))

# Mostrar resumen
model.summary()

# ----------------------------------------------------------------
# PASO 3: Compilación y Entrenamiento con Generadores
# ----------------------------------------------------------------
optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Callback para reducir learning rate de forma más agresiva
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,              # Reduce LR más agresivamente
    patience=5,              # Más paciencia para permitir convergencia
    min_lr=1e-8,
    verbose=1
)

# ELIMINAMOS Early Stopping para que entrene las 35 épocas completas
# Si quieres usar Early Stopping más tarde, úsalo con patience=15 o más
print("ENTRENAMIENTO CONFIGURADO PARA 35 ÉPOCAS COMPLETAS (sin Early Stopping)")

print("\nIniciando entrenamiento con generadores...")
print(f"Steps per epoch calculados: {train_generator.samples // BATCH_SIZE}")
print(f"Validation steps calculados: {validation_generator.samples // BATCH_SIZE}")

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[reduce_lr],  # Solo reduce_lr, sin early_stop
    # use_multiprocessing=True,  # Comentado para evitar problemas en Windows
    # workers=4  # Comentado temporalmente
)


print(f"\nMejor precisión en validación: {max(history.history['val_accuracy'])*100:.2f}%")

# ----------------------------------------------------------------
# PASO 5: Guardado del Modelo
# ----------------------------------------------------------------
model_filename = "animal_classifier_optimized.h5"
model.save(model_filename)
print(f"\nModelo guardado exitosamente como: {model_filename}")

# ----------------------------------------------------------------
# PASO 6: Visualización de Resultados
# ----------------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()