import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ==========================================
# SETTINGS
# ==========================================

DATASET_DIR = "dataset"

LABELS = [
    "HELLO",
    "YES"
]

SEQUENCE_LENGTH = 30
FEATURES = 63


# ==========================================
# LOAD DATASET
# ==========================================

X = []
y = []

print("\nLoading dataset...\n")

for label in LABELS:

    folder = os.path.join(
        DATASET_DIR,
        label
    )

    if not os.path.exists(folder):
        continue

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ]

    for file in files:

        path = os.path.join(
            folder,
            file
        )

        sequence = np.load(path)

        # Make sure shape is correct
        if sequence.shape != (SEQUENCE_LENGTH, FEATURES):
            print(
                "Skipping:",
                file,
                sequence.shape
            )
            continue

        X.append(sequence)
        y.append(label)


X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("Number of samples:", len(X))


# ==========================================
# ENCODE LABELS
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nClasses:")
print(encoder.classes_)

y_encoded = to_categorical(
    y_encoded
)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# BUILD LSTM
# ==========================================

model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            SEQUENCE_LENGTH,
            FEATURES
        )
    ),

    Dropout(0.3),

    LSTM(64),

    Dropout(0.3),

    Dense(32, activation="relu"),

    Dense(
        len(LABELS),
        activation="softmax"
    )
])


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# MODEL SUMMARY
# ==========================================

print("\n==============================")
print("MODEL")
print("==============================")

model.summary()


# ==========================================
# TRAIN
# ==========================================

print("\n==============================")
print("TRAINING")
print("==============================")

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=8,
    validation_data=(
        X_test,
        y_test
    ),
    verbose=1
)


# ==========================================
# EVALUATE
# ==========================================

print("\n==============================")
print("EVALUATION")
print("==============================")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# SAVE MODEL
# ==========================================

model.save(
    "sign_model.keras"
)

np.save(
    "label_classes.npy",
    encoder.classes_
)

print("\n==============================")
print("MODEL SAVED")
print("==============================")

print("sign_model.keras")
print("label_classes.npy")