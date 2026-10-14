from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.svm import SVC


#Xác định thư mục gốc của dự án.
PROJECT_DIR = Path(__file__).resolve().parent.parent

#Thư mục chứa dữ liệu đặc trưng đã được chuẩn bị.
DATA_DIR = PROJECT_DIR / "data" / "svm_data"

#Thư mục lưu mô hình đã huấn luyện.
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "svm_model.pkl"

#Thư mục lưu kết quả đánh giá.
RESULTS_DIR = PROJECT_DIR / "results" / "svm"

#Thứ tự lớp phải thống nhất với svm_prepare.py và svm_predict.py.
CLASSES = ["forest", "lake", "residential", "river", "road"]

#Tham số của mô hình SVM.
SVM_PARAMS = {
    "kernel": "rbf",
    "C": 10,
    "gamma": "scale",
    "class_weight": "balanced",
    "decision_function_shape": "ovr",
}


def load_data():
    """Đọc dữ liệu train và validation từ các tệp .npy."""
    required_files = [
        DATA_DIR / "X_train.npy",
        DATA_DIR / "y_train.npy",
        DATA_DIR / "X_val.npy",
        DATA_DIR / "y_val.npy",
    ]

    #Kiểm tra các tệp dữ liệu có tồn tại hay không.
    missing_files = [
        path for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        missing_list = "\n".join(
            f"  - {path}" for path in missing_files
        )
        raise FileNotFoundError(
            "Không tìm thấy các tệp dữ liệu sau:\n"
            f"{missing_list}\n"
            "Vui lòng chạy src/svm_prepare.py trước."
        )

    print("\nĐang tải dữ liệu huấn luyện và validation...")

    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")

    #Kiểm tra dữ liệu có rỗng hay không.
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Dữ liệu train đang rỗng.")

    if X_val.size == 0 or y_val.size == 0:
        raise ValueError("Dữ liệu validation đang rỗng.")

    #Kiểm tra số lượng mẫu và nhãn có khớp nhau không.
    if len(X_train) != len(y_train):
        raise ValueError(
            "Số lượng mẫu X_train và y_train không khớp."
        )

    if len(X_val) != len(y_val):
        raise ValueError(
            "Số lượng mẫu X_val và y_val không khớp."
        )

    #Kiểm tra số chiều đặc trưng giữa train và validation.
    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError(
            "X_train và X_val phải là ma trận hai chiều."
        )

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            "Số chiều đặc trưng của train và validation không khớp."
        )

    #Kiểm tra giá trị NaN hoặc vô cực.
    if not np.isfinite(X_train).all():
        raise ValueError("X_train chứa NaN hoặc giá trị vô cực.")

    if not np.isfinite(X_val).all():
        raise ValueError("X_val chứa NaN hoặc giá trị vô cực.")

    #Kiểm tra nhãn nằm trong phạm vi các lớp đã cấu hình.
    valid_labels = set(range(len(CLASSES)))

    if not set(np.unique(y_train)).issubset(valid_labels):
        raise ValueError("y_train chứa nhãn không hợp lệ.")

    if not set(np.unique(y_val)).issubset(valid_labels):
        raise ValueError("y_val chứa nhãn không hợp lệ.")

    print("Đã tải dữ liệu thành công.")
    print(f"  - Train: X = {X_train.shape}, y = {y_train.shape}")
    print(f"  - Validation: X = {X_val.shape}, y = {y_val.shape}")

    return X_train, y_train, X_val, y_val


def build_model():
    """Khởi tạo mô hình SVM theo cấu hình đã thiết lập."""
    print("\nCấu hình mô hình SVM:")
    print(f"  - Kernel: {SVM_PARAMS['kernel']}")
    print(f"  - C: {SVM_PARAMS['C']}")
    print(f"  - Gamma: {SVM_PARAMS['gamma']}")
    print(f"  - Chiến lược đa lớp: One-vs-Rest (OvR)")
    print(
        f"  - Trọng số lớp: "
        f"{SVM_PARAMS['class_weight']}"
    )

    return SVC(**SVM_PARAMS)


def train_model(model, X_train, y_train):
    """Huấn luyện SVM trên tập dữ liệu train."""
    print("\nĐang huấn luyện mô hình SVM...")
    print("Quá trình này có thể mất thời gian tùy kích thước dữ liệu.")

    model.fit(X_train, y_train)

    print("Huấn luyện mô hình hoàn tất.")

    return model


def evaluate_model(model, X_val, y_val):
    """Đánh giá mô hình trên tập validation."""
    print("\nĐang đánh giá mô hình trên tập validation...")

    y_pred = model.predict(X_val)

    #Tính các chỉ số đánh giá.
    accuracy = accuracy_score(y_val, y_pred)

    precision_macro = precision_score(
        y_val,
        y_pred,
        labels=list(range(len(CLASSES))),
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_val,
        y_pred,
        labels=list(range(len(CLASSES))),
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_val,
        y_pred,
        labels=list(range(len(CLASSES))),
        average="macro",
        zero_division=0,
    )

    precision_weighted = precision_score(
        y_val,
        y_pred,
        labels=list(range(len(CLASSES))),
        average="weighted",
        zero_division=0,
    )

    recall_weighted = recall_score(
        y_val,
        y_pred,
        labels=list(range(len(CLASSES))),
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_val,
        y_pred,
        labels=list(range(len(CLASSES))),
        average="weighted",
        zero_division=0,
    )

    #Báo cáo chi tiết cho từng lớp.
    report = classification_report(
        y_val,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )

    #Ma trận nhầm lẫn: hàng là nhãn thật, cột là nhãn dự đoán.
    matrix = confusion_matrix(
        y_val,
        y_pred,
        labels=list(range(len(CLASSES))),
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }

    print("\n" + "=" * 65)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH SVM")
    print("=" * 65)

    print(f"  - Accuracy:          {accuracy:.4f}")
    print(f"  - Precision (macro): {precision_macro:.4f}")
    print(f"  - Recall (macro):    {recall_macro:.4f}")
    print(f"  - F1-score (macro):  {f1_macro:.4f}")
    print(f"  - Precision (weighted): {precision_weighted:.4f}")
    print(f"  - Recall (weighted):    {recall_weighted:.4f}")
    print(f"  - F1-score (weighted):  {f1_weighted:.4f}")

    print("\nBáo cáo phân loại theo từng lớp:")
    print(report)

    print("Ma trận nhầm lẫn:")
    print(matrix)

    return metrics, report, matrix


def save_results(model, metrics, report, matrix):
    """Lưu mô hình SVM và các kết quả đánh giá vào thư mục dự án."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    #Lưu mô hình đã huấn luyện.
    joblib.dump(model, MODEL_PATH)

    #Lưu báo cáo phân loại dưới dạng văn bản.
    report_path = RESULTS_DIR / "classification_report.txt"
    report_path.write_text(
        report,
        encoding="utf-8",
    )

    #Lưu ma trận nhầm lẫn dưới dạng CSV.
    matrix_path = RESULTS_DIR / "confusion_matrix.csv"

    with matrix_path.open("w", encoding="utf-8-sig") as file:
        file.write("Nhãn thật/Nhãn dự đoán," + ",".join(CLASSES) + "\n")

        for class_name, row in zip(CLASSES, matrix):
            file.write(
                class_name + "," +
                ",".join(str(int(value)) for value in row) +
                "\n"
            )

    #Lưu các chỉ số tổng hợp dưới dạng JSON.
    metrics_path = RESULTS_DIR / "metrics_summary.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\nĐã lưu mô hình và kết quả:")
    print(f"  - Mô hình SVM: {MODEL_PATH}")
    print(f"  - Báo cáo phân loại: {report_path}")
    print(f"  - Ma trận nhầm lẫn: {matrix_path}")
    print(f"  - Chỉ số đánh giá: {metrics_path}")


def main():
    """Điều phối các bước tải dữ liệu, huấn luyện, đánh giá và lưu."""
    print("=" * 65)
    print("HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH SVM")
    print("=" * 65)

    X_train, y_train, X_val, y_val = load_data()

    model = build_model()

    model = train_model(model, X_train, y_train)

    metrics, report, matrix = evaluate_model(
        model,
        X_val,
        y_val,
    )

    save_results(model, metrics, report, matrix)

    print("\n" + "=" * 65)
    print("HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN SVM!")
    print("=" * 65)


if __name__ == "__main__":
    main()