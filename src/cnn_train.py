from pathlib import Path
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#Xác định thư mục gốc của project
PROJECT_DIR = Path(__file__).resolve().parent.parent

#Đường dẫn dữ liệu đã xử lý
DATA_DIR = PROJECT_DIR / "data" / "data_processed"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

#Thư mục lưu mô hình
MODEL_DIR = PROJECT_DIR / "models"

BEST_MODEL_PATH = MODEL_DIR / "cnn_model_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "cnn_model.keras"


CLASSES = ["forest", "lake", "residential", "river", "road"]

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42


def set_random_seed():
    """Thiết lập seed để tăng khả năng tái lập kết quả."""
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def validate_dataset():
    """Kiểm tra cấu trúc dữ liệu train và validation."""
    for split_dir in [TRAIN_DIR, VAL_DIR]:
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục dữ liệu: {split_dir}\n"
                "Hãy chạy clean_dataset.py trước."
            )

        for cls in CLASSES:
            class_dir = split_dir / cls

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Không tìm thấy thư mục lớp dữ liệu: {class_dir}"
                )

            image_count = sum(
                1 for path in class_dir.iterdir()
                if path.is_file()
            )

            if image_count == 0:
                raise ValueError(
                    f"Thư mục không có ảnh: {class_dir}"
                )

    print("Đã kiểm tra cấu trúc dữ liệu.")
    print(f"Train      : {TRAIN_DIR}")
    print(f"Validation : {VAL_DIR}")


def prepare_data_generators():
    #Augmentation cho tập train
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=90,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.0,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest"
    )

    #Chỉ chuẩn hóa dữ liệu validation
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    #Generator cho tập train
    train_generator = train_datagen.flow_from_directory(
        directory=str(TRAIN_DIR),
        classes=CLASSES,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True,
        seed=SEED
    )

    #Generator cho tập validation
    validation_generator = validation_datagen.flow_from_directory(
        directory=str(VAL_DIR),
        classes=CLASSES,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False
    )

    #Kiểm tra số lớp và thứ tự nhãn
    expected_class_indices = {
        cls: index for index, cls in enumerate(CLASSES)
    }

    if train_generator.class_indices != expected_class_indices:
        raise ValueError(
            "Thứ tự nhãn của tập train không đúng."
        )

    if validation_generator.class_indices != expected_class_indices:
        raise ValueError(
            "Thứ tự nhãn của tập validation không đúng."
        )

    print("\nSố lượng ảnh:")
    print(f"Train      : {train_generator.samples}")
    print(f"Validation : {validation_generator.samples}")
    print(f"Số lớp     : {train_generator.num_classes}")
    print(f"Nhãn lớp   : {train_generator.class_indices}")

    return train_generator, validation_generator


def build_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),

        #BLOCK 1
        layers.Conv2D(
            32, (3, 3), activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        #BLOCK 2
        layers.Conv2D(
            64, (3, 3), activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        #BLOCK 3
        layers.Conv2D(
            128, (3, 3), activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            128, (3, 3), activation="relu"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        #CLASSIFIER
        layers.Flatten(),

        layers.Dropout(0.3),

        layers.Dense(
            512, activation="relu"
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.6),

        layers.Dense(
            len(CLASSES),
            activation="softmax"
        )
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        metrics=["accuracy"]
    )

    return model


def prepare_callbacks():
    """Tạo callbacks để dừng sớm và lưu mô hình tốt nhất."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    return [early_stop, checkpoint]


def train_model(model, train_generator, validation_generator):
    """Huấn luyện CNN và lưu mô hình sau khi hoàn tất."""
    callbacks = prepare_callbacks()

    print("\nBắt đầu huấn luyện CNN...")

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )

    #Lưu mô hình với trọng số tốt nhất đã được EarlyStopping khôi phục
    model.save(str(FINAL_MODEL_PATH))

    print("\nĐã hoàn tất huấn luyện CNN.")
    print(f"Mô hình tốt nhất : {BEST_MODEL_PATH}")
    print(f"Mô hình cuối cùng: {FINAL_MODEL_PATH}")

    return history


def main():
    """Thực hiện toàn bộ quy trình huấn luyện CNN."""
    print("=" * 60)
    print("HUẤN LUYỆN CNN - SATELLITE IMAGE CLASSIFICATION")
    print("=" * 60)

    #Thiết lập seed
    set_random_seed()

    #Kiểm tra dữ liệu
    validate_dataset()

    #Chuẩn bị dữ liệu
    train_generator, validation_generator = (
        prepare_data_generators()
    )

    #Xây dựng mô hình
    model = build_cnn_model()

    print("\nKiến trúc mô hình CNN:")
    model.summary()

    #Huấn luyện và lưu mô hình
    history = train_model(
        model,
        train_generator,
        validation_generator
    )

    print("\nQuá trình huấn luyện đã hoàn tất!")

    return history


if __name__ == "__main__":
    main()