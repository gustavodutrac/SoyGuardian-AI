import tensorflow as tf
import numpy as np

from keras.applications import EfficientNetB0
from keras import layers
from keras import Model
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ==================================
# CONFIGURAÇÕES
# ==================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

# ==================================
# DATASET
# ==================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    "soybean.leaf.dataset",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "soybean.leaf.dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("\nClasses encontradas:")
print(class_names)

# ==================================
# CONTAGEM DE IMAGENS
# ==================================

counts = {
    "Caterpillar": 3309,
    "Diabrotica speciosa": 2205,
    "Healthy": 896
}

total = sum(counts.values())

class_weight = {
    0: total / (3 * counts["Caterpillar"]),
    1: total / (3 * counts["Diabrotica speciosa"]),
    2: total / (3 * counts["Healthy"])
}

print("\nPesos utilizados:")
print(class_weight)

# ==================================
# PERFORMANCE
# ==================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)

# ==================================
# DATA AUGMENTATION
# ==================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.15),

    layers.RandomZoom(0.15),

    layers.RandomContrast(0.2)

])

# ==================================
# MODELO BASE
# ==================================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)

base_model.trainable = False

# ==================================
# ARQUITETURA
# ==================================

inputs = tf.keras.Input(
    shape=(224,224,3)
)

x = data_augmentation(inputs)

x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = Model(
    inputs,
    outputs
)

# ==================================
# TREINO INICIAL
# ==================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [

    EarlyStopping(
        patience=4,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "models/soybean_model.keras",
        save_best_only=True
    )

]

print("\nIniciando Treinamento 1...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weight
)

# ==================================
# FINE TUNING
# ==================================

print("\nIniciando Fine Tuning...")

base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weight,
    callbacks=callbacks
)

# ==================================
# SALVAR MODELO
# ==================================

model.save(
    "models/soybean_model_final.keras"
)

print("\nTreinamento concluído!")

print(
    "Modelo salvo em models/soybean_model_final.keras"
)