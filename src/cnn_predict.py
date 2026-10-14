from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


#Xác định thư mục gốc của project
PROJECT_DIR = Path(__file__).resolve().parent.parent

#Đường dẫn mô hình đã huấn luyện
MODEL_PATH = PROJECT_DIR / "models" / "cnn_model.keras"

#Đường dẫn dữ liệu test
TEST_DIR = PROJECT_DIR / "data" / "data_processed" / "test"

#Thư mục lưu kết quả đánh giá
RESULT_DIR = PROJECT_DIR / "results" / "cnn"

PREDICTIONS_FILE = RESULT_DIR / "predictions.csv"
REPORT_FILE = RESULT_DIR / "classification_report.csv"
CONFUSION_MATRIX_FILE = RESULT_DIR / "confusion_matrix.csv"
METRICS_FILE = RESULT_DIR / "metrics_summary.json"


CLASSES = ["forest", "lake", "residential", "river", "road"]

IMG_SIZE = (128, 128)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_paths():
    """Kiểm tra mô hình và thư mục dữ liệu test."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình CNN: {MODEL_PATH}\n"
            "Hãy chạy cnn_train.py trước."
        )

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục test: {TEST_DIR}\n"
            "Hãy chạy clean_dataset.py trước."
        )

    for cls in CLASSES:
        class_dir = TEST_DIR / cls

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục lớp test: {class_dir}"
            )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def collect_test_images():
    """Thu thập ảnh test và nhãn thực tế từ tên thư mục lớp."""
    test_images = []

    for cls in CLASSES:
        class_dir = TEST_DIR / cls

        image_paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        for img_path in image_paths:
            test_images.append({
                "path": img_path,
                "actual_label": cls
            })

        print(f"{cls}: {len(image_paths)} ảnh test")

    if not test_images:
        raise ValueError(
            f"Không tìm thấy ảnh hợp lệ trong: {TEST_DIR}"
        )

    print(f"\nTổng số ảnh test tìm thấy: {len(test_images)}")

    return test_images


def predict_image(model, img_path):
    """Dự đoán lớp ảnh và xác suất của từng lớp."""

    img = image.load_img(
        str(img_path),
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    probabilities = model.predict(
        img_array,
        verbose=0
    )[0]

    if len(probabilities) != len(CLASSES):
        raise ValueError(
            f"Mô hình trả về {len(probabilities)} đầu ra, "
            f"nhưng dự án có {len(CLASSES)} lớp."
        )

    predicted_idx = int(np.argmax(probabilities))
    predicted_label = CLASSES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    return predicted_label, confidence, probabilities


def run_predictions(model, test_images):
    """Dự đoán toàn bộ ảnh test và lưu kết quả từng ảnh."""
    results = []
    skipped_images = 0

    total_images = len(test_images)

    for index, item in enumerate(test_images, start=1):
        img_path = item["path"]
        actual_label = item["actual_label"]

        try:
            predicted_label, confidence, probabilities = (
                predict_image(model, img_path)
            )

        except Exception as error:
            skipped_images += 1

            print(
                f"[LỖI] Không xử lý được ảnh: {img_path} | "
                f"{error}"
            )
            continue

        is_correct = predicted_label == actual_label

        row = {
            "file_name": img_path.name,
            "image_path": img_path.relative_to(
                PROJECT_DIR
            ).as_posix(),
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 6),
            "confidence_percent": round(
                confidence * 100, 2
            ),
            "is_correct": is_correct
        }

        #Lưu xác suất dự đoán của cả 5 lớp
        for class_idx, cls in enumerate(CLASSES):
            row[f"prob_{cls}"] = round(
                float(probabilities[class_idx]), 6
            )

        results.append(row)

        if not is_correct:
            print(
                f"[SAI] {img_path.name} | "
                f"Thực tế: {actual_label} | "
                f"Dự đoán: {predicted_label} | "
                f"Độ tin cậy: {confidence * 100:.2f}%"
            )

        if index % 100 == 0 or index == total_images:
            print(
                f"Tiến trình: {index}/{total_images} ảnh"
            )

    if not results:
        raise ValueError(
            "Không có ảnh nào được dự đoán thành công."
        )

    predictions_df = pd.DataFrame(results)

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nĐã lưu dự đoán tại: {PREDICTIONS_FILE}")
    print(f"Số ảnh dự đoán thành công: {len(results)}")
    print(f"Số ảnh bị bỏ qua: {skipped_images}")

    return predictions_df


def evaluate_model(predictions_df):
    """Tính các chỉ số đánh giá và lưu báo cáo."""
    y_true = predictions_df["actual_label"].tolist()
    y_pred = predictions_df["predicted_label"].tolist()

    #Accuracy tổng thể
    accuracy = accuracy_score(y_true, y_pred)

    #Macro: coi trọng các lớp như nhau
    precision_macro = precision_score(
        y_true, y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true, y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true, y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0
    )

    #Weighted: tính đến số lượng ảnh mỗi lớp
    precision_weighted = precision_score(
        y_true, y_pred,
        labels=CLASSES,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_true, y_pred,
        labels=CLASSES,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true, y_pred,
        labels=CLASSES,
        average="weighted",
        zero_division=0
    )

    #Báo cáo chi tiết từng lớp
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()

    report_df.to_csv(
        REPORT_FILE,
        encoding="utf-8-sig"
    )

    #Ma trận nhầm lẫn
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES
    )

    cm_df.index.name = "actual_label"
    cm_df.columns.name = "predicted_label"

    cm_df.to_csv(
        CONFUSION_MATRIX_FILE,
        encoding="utf-8-sig"
    )

    #Tổng hợp các chỉ số chính
    metrics = {
        "model": "CNN",
        "test_directory": str(TEST_DIR),
        "num_test_images": int(len(predictions_df)),
        "num_correct": int(
            predictions_df["is_correct"].sum()
        ),
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "classes": CLASSES
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4
        )

    #In kết quả ra terminal
    print("\n" + "=" * 65)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH CNN")
    print("=" * 65)

    print(f"Số ảnh test       : {len(predictions_df)}")
    print(f"Số ảnh dự đoán đúng: {metrics['num_correct']}")
    print(f"Accuracy           : {accuracy:.4f}")

    print("\n--- MACRO AVERAGE ---")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall   : {recall_macro:.4f}")
    print(f"F1-score : {f1_macro:.4f}")

    print("\n--- WEIGHTED AVERAGE ---")
    print(f"Precision: {precision_weighted:.4f}")
    print(f"Recall   : {recall_weighted:.4f}")
    print(f"F1-score : {f1_weighted:.4f}")

    print("\n--- CLASSIFICATION REPORT ---")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASSES,
            target_names=CLASSES,
            digits=4,
            zero_division=0
        )
    )

    print("--- CONFUSION MATRIX ---")
    print("Hàng = nhãn thực tế; cột = nhãn dự đoán")
    print(cm_df)

    print("\nCác file kết quả đã lưu:")
    print(f"Predictions : {PREDICTIONS_FILE}")
    print(f"Report      : {REPORT_FILE}")
    print(f"Confusion   : {CONFUSION_MATRIX_FILE}")
    print(f"Metrics     : {METRICS_FILE}")

    return metrics


def main():
    print("=" * 65)
    print("CNN PREDICTION - SATELLITE IMAGE CLASSIFICATION")
    print("=" * 65)

    #Kiểm tra đường dẫn
    validate_paths()

    #Tải mô hình đã huấn luyện
    print(f"\nĐang tải mô hình: {MODEL_PATH}")

    model = load_model(str(MODEL_PATH))

    print("Đã tải mô hình thành công.")

    #Thu thập dữ liệu test
    test_images = collect_test_images()

    #Dự đoán và lưu kết quả
    predictions_df = run_predictions(
        model,
        test_images
    )

    #Đánh giá mô hình
    evaluate_model(predictions_df)

    print("\nHoàn tất dự đoán và đánh giá CNN!")


if __name__ == "__main__":
    main()