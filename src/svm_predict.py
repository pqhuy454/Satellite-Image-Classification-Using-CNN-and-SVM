from pathlib import Path
import json

import cv2
import joblib
import numpy as np
import pandas as pd

from skimage.feature import local_binary_pattern, hog

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


#Xác định thư mục gốc của dự án.
PROJECT_DIR = Path(__file__).resolve().parent.parent

#Thư mục chứa ảnh test đã được làm sạch và phân lớp.
TEST_DIR = PROJECT_DIR / "data" / "data_processed" / "test"

#Đường dẫn mô hình SVM và bộ chuẩn hóa đặc trưng.
MODEL_PATH = PROJECT_DIR / "models" / "svm_model.pkl"
SCALER_PATH = PROJECT_DIR / "data" / "svm_data" / "scaler.pkl"

#Thư mục lưu kết quả dự đoán và đánh giá.
OUTPUT_DIR = PROJECT_DIR / "results_csv"

#Danh sách lớp và thứ tự nhãn phải khớp với các file huấn luyện.
CLASSES = ["forest", "lake", "residential", "river", "road"]

#Các định dạng ảnh được chấp nhận.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#Cấu hình LBP đa tỉ lệ.
PATCH_GRID = 4

LBP_CONFIGS = [
    (1, 8),
    (2, 16),
]

LBP_METHOD = "uniform"

#Cấu hình HOG phải giống svm_prepare.py.
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}

#Kích thước ảnh sử dụng khi trích xuất đặc trưng.
IMAGE_SIZE = (128, 128)


def lbp_histogram(gray_patch, radius, points):
    """Tính histogram LBP cho một vùng ảnh xám."""
    lbp_image = local_binary_pattern(
        gray_patch,
        P=points,
        R=radius,
        method=LBP_METHOD,
    )

    number_of_bins = points + 2

    histogram, _ = np.histogram(
        lbp_image.ravel(),
        bins=np.arange(number_of_bins + 1),
        range=(0, number_of_bins),
    )

    histogram = histogram.astype(np.float32)
    histogram /= histogram.sum() + 1e-6

    return histogram


def extract_lbp_patch(gray_image):
    """Chia ảnh thành lưới PATCH_GRID x PATCH_GRID."""
    height, width = gray_image.shape

    patch_height = height // PATCH_GRID
    patch_width = width // PATCH_GRID

    if patch_height == 0 or patch_width == 0:
        raise ValueError("Ảnh quá nhỏ để trích xuất đặc trưng LBP.")

    features = []

    for row in range(PATCH_GRID):
        for col in range(PATCH_GRID):
            #Xác định tọa độ patch, bao gồm cả phần dư kích thước.
            y_start = row * patch_height
            x_start = col * patch_width

            y_end = (
                (row + 1) * patch_height
                if row < PATCH_GRID - 1
                else height
            )

            x_end = (
                (col + 1) * patch_width
                if col < PATCH_GRID - 1
                else width
            )

            patch = gray_image[y_start:y_end, x_start:x_end]

            #Trích xuất LBP cho tất cả cấu hình đã thiết lập.
            for radius, points in LBP_CONFIGS:
                histogram = lbp_histogram(
                    patch,
                    radius=radius,
                    points=points,
                )
                features.extend(histogram)

    return np.asarray(features, dtype=np.float32)


def extract_hog(gray_image):
    """Trích xuất HOG để mô tả hướng gradient và cấu trúc ảnh."""
    return hog(
        gray_image,
        orientations=HOG_PARAMS["orientations"],
        pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
        cells_per_block=HOG_PARAMS["cells_per_block"],
        block_norm=HOG_PARAMS["block_norm"],
        feature_vector=True,
    ).astype(np.float32)


def extract_features(image_path):
    """Đọc ảnh, chuyển sang ảnh xám, chuẩn hóa kích thước và ghép đặc trưng LBP với HOG."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    #Chuyển ảnh màu sang ảnh xám.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Đưa ảnh về cùng kích thước với dữ liệu lúc huấn luyện.
    gray_image = cv2.resize(
        gray_image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    #Trích xuất hai nhóm đặc trưng.
    lbp_features = extract_lbp_patch(gray_image)
    hog_features = extract_hog(gray_image)

    #Ghép thành vector đặc trưng duy nhất.
    features = np.concatenate(
        [lbp_features, hog_features]
    )

    return features.astype(np.float32)


def get_image_paths(class_dir):
    """Lấy danh sách ảnh trong thư mục lớp và sắp xếp ổn định."""
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_test_data():
    """Đọc toàn bộ ảnh test theo cấu trúc thư mục."""
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục test: {TEST_DIR}"
        )

    features_list = []
    labels_list = []
    image_paths = []
    true_labels = []

    #Thống kê số ảnh hợp lệ của từng lớp.
    class_counts = {}

    print("\nĐang đọc ảnh và trích xuất đặc trưng tập test...")

    for label, class_name in enumerate(CLASSES):
        class_dir = TEST_DIR / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục lớp: {class_dir}"
            )

        paths = get_image_paths(class_dir)

        if not paths:
            print(
                f"  Cảnh báo: Lớp '{class_name}' "
                "không có ảnh trong tập test."
            )

        successful_count = 0

        print(
            f"\nLớp '{class_name}': "
            f"{len(paths)} ảnh được tìm thấy."
        )

        for index, image_path in enumerate(paths, start=1):
            try:
                features = extract_features(image_path)

                features_list.append(features)
                labels_list.append(label)
                image_paths.append(str(image_path.relative_to(PROJECT_DIR)))
                true_labels.append(class_name)

                successful_count += 1

            except (ValueError, cv2.error, OSError) as error:
                #Chỉ bỏ qua ảnh không đọc hoặc không xử lý được.
                print(
                    f"  Cảnh báo: Bỏ qua ảnh lỗi "
                    f"'{image_path.name}': {error}"
                )

            #Thông báo tiến độ định kỳ với tập dữ liệu lớn.
            if index % 100 == 0 or index == len(paths):
                print(
                    f"  Tiến độ: {index}/{len(paths)} ảnh đã xử lý."
                )

        class_counts[class_name] = successful_count

        print(
            f"  Số ảnh hợp lệ: "
            f"{successful_count}/{len(paths)}"
        )

    if not features_list:
        raise ValueError(
            "Không có ảnh hợp lệ để đánh giá. "
            "Hãy kiểm tra dữ liệu trong thư mục test."
        )

    X_test = np.vstack(features_list).astype(np.float32)
    y_test = np.asarray(labels_list, dtype=np.int64)

    print("\nThống kê số ảnh test hợp lệ:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} ảnh")

    print(f"\nTổng số ảnh test hợp lệ: {len(y_test)}")
    print(f"Kích thước ma trận đặc trưng: {X_test.shape}")

    return X_test, y_test, image_paths, true_labels


def evaluate_model(y_true, y_pred):
    """Tính Accuracy, Precision, Recall, F1-score, báo cáo phân loại và ma trận nhầm lẫn."""
    class_labels = list(range(len(CLASSES)))

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(
        y_true,
        y_pred,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )

    precision_weighted = precision_score(
        y_true,
        y_pred,
        labels=class_labels,
        average="weighted",
        zero_division=0,
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        labels=class_labels,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        labels=class_labels,
        average="weighted",
        zero_division=0,
    )

    #Báo cáo giữ đủ năm lớp, kể cả lớp không có mẫu trong test.
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )

    #Hàng là nhãn thật; cột là nhãn dự đoán.
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_labels,
    )

    metrics = {
        "model": "SVM",
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }

    return metrics, report_dict, report_text, matrix


def save_results(
    image_paths,
    y_true,
    y_pred,
    true_labels,
    metrics,
    report_dict,
    report_text,
    matrix,
):
    """Lưu dự đoán từng ảnh, chỉ số theo lớp, tổng hợp mô hình và ma trận nhầm lẫn để sử dụng trong svm_visualize.py."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_predictions = pd.DataFrame({
        "image_path": image_paths,
        "y_true": y_true,
        "y_pred": y_pred,
        "true_label": true_labels,
        "pred_label": [CLASSES[index] for index in y_pred],
        "is_correct": (y_true == y_pred),
    })

    predictions_path = OUTPUT_DIR / "svm_predictions.csv"

    df_predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    rows = []

    for class_name in CLASSES:
        class_metrics = report_dict[class_name]

        rows.append({
            "class": class_name,
            "precision": class_metrics["precision"],
            "recall": class_metrics["recall"],
            "f1_score": class_metrics["f1-score"],
            "support": int(class_metrics["support"]),
        })

    df_metrics = pd.DataFrame(rows)

    metrics_path = OUTPUT_DIR / "svm_metrics.csv"

    df_metrics.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_path = OUTPUT_DIR / "svm_summary.csv"

    pd.DataFrame([metrics]).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    df_confusion = pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES,
    )

    df_confusion.index.name = "true_label"

    confusion_path = OUTPUT_DIR / "svm_confusion_matrix.csv"

    df_confusion.to_csv(
        confusion_path,
        encoding="utf-8-sig",
    )

    df_report = pd.DataFrame(report_dict).transpose()

    report_path = OUTPUT_DIR / "svm_classification_report.csv"

    df_report.to_csv(
        report_path,
        encoding="utf-8-sig",
    )

    summary_json_path = OUTPUT_DIR / "svm_metrics_summary.json"

    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\nĐã lưu các tệp kết quả:")
    print(f"  - Dự đoán từng ảnh: {predictions_path}")
    print(f"  - Chỉ số theo lớp: {metrics_path}")
    print(f"  - Tổng hợp mô hình: {summary_path}")
    print(f"  - Ma trận nhầm lẫn: {confusion_path}")
    print(f"  - Báo cáo phân loại: {report_path}")
    print(f"  - Chỉ số JSON: {summary_json_path}")


def main():
    """Thực hiện toàn bộ quy trình dự đoán và đánh giá SVM trên test."""
    print("=" * 70)
    print("DỰ ĐOÁN VÀ ĐÁNH GIÁ MÔ HÌNH SVM TRÊN TẬP TEST")
    print("=" * 70)

    print(f"\nThư mục dự án: {PROJECT_DIR}")
    print(f"Thư mục test: {TEST_DIR}")
    print(f"Đường dẫn mô hình: {MODEL_PATH}")
    print(f"Đường dẫn scaler: {SCALER_PATH}")
    print(f"Thư mục lưu kết quả: {OUTPUT_DIR}")

    print("\n[BƯỚC 1/5] TẢI MÔ HÌNH VÀ SCALER")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình SVM: {MODEL_PATH}\n"
            "Vui lòng chạy src/svm_train.py trước."
        )

    if not SCALER_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy scaler: {SCALER_PATH}\n"
            "Vui lòng chạy src/svm_prepare.py trước."
        )

    svm = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("Đã tải mô hình SVM và scaler thành công.")

    print("\n[BƯỚC 2/5] ĐỌC DỮ LIỆU TEST")

    X_test, y_test, image_paths, true_labels = load_test_data()

    #Kiểm tra số chiều đặc trưng trước khi chuẩn hóa.
    if X_test.shape[1] != scaler.n_features_in_:
        raise ValueError(
            "Số chiều đặc trưng ảnh test không khớp scaler.\n"
            f"Đặc trưng test: {X_test.shape[1]}\n"
            f"Scaler yêu cầu: {scaler.n_features_in_}"
        )

    print("\n[BƯỚC 3/5] CHUẨN HÓA ĐẶC TRƯNG")

    X_test = scaler.transform(X_test).astype(np.float32)

    print("Đã chuẩn hóa đặc trưng test thành công.")
    print(f"Kích thước dữ liệu sau chuẩn hóa: {X_test.shape}")

    print("\n[BƯỚC 4/5] DỰ ĐOÁN VÀ ĐÁNH GIÁ")

    y_pred = svm.predict(X_test)

    metrics, report_dict, report_text, matrix = evaluate_model(
        y_true=y_test,
        y_pred=y_pred,
    )

    print("\nKẾT QUẢ ĐÁNH GIÁ:")
    print(f"  - Số ảnh test: {metrics['test_samples']}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  - Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"  - F1-score (macro): {metrics['f1_macro']:.4f}")
    print(
        f"  - Precision (weighted): "
        f"{metrics['precision_weighted']:.4f}"
    )
    print(
        f"  - Recall (weighted): "
        f"{metrics['recall_weighted']:.4f}"
    )
    print(
        f"  - F1-score (weighted): "
        f"{metrics['f1_weighted']:.4f}"
    )

    print("\nBÁO CÁO PHÂN LOẠI:")
    print(report_text)

    print("MA TRẬN NHẦM LẪN:")
    print("Hàng: nhãn thật | Cột: nhãn dự đoán")
    print(matrix)

    print("\n[BƯỚC 5/5] LƯU KẾT QUẢ CHO TRỰC QUAN HÓA")

    save_results(
        image_paths=image_paths,
        y_true=y_test,
        y_pred=y_pred,
        true_labels=true_labels,
        metrics=metrics,
        report_dict=report_dict,
        report_text=report_text,
        matrix=matrix,
    )

    print("\n" + "=" * 70)
    print("HOÀN TẤT DỰ ĐOÁN VÀ ĐÁNH GIÁ MÔ HÌNH SVM!")
    print("=" * 70)


if __name__ == "__main__":
    main()