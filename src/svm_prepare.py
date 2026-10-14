from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern, hog
from sklearn.preprocessing import StandardScaler


#Xác định thư mục gốc của dự án từ vị trí tệp hiện tại.
PROJECT_DIR = Path(__file__).resolve().parent.parent

#Thư mục chứa dữ liệu ảnh đã được làm sạch và chia tập.
DATA_DIR = PROJECT_DIR / "data" / "data_processed"

#Thư mục lưu dữ liệu đặc trưng phục vụ huấn luyện SVM.
OUTPUT_DIR = PROJECT_DIR / "data"/ "svm_data"

#Danh sách lớp ảnh, thứ tự nhãn phải thống nhất trong toàn bộ dự án.
CLASSES = ["forest", "lake", "residential", "river", "road"]

#Chia ảnh thành lưới 4 x 4 để trích xuất đặc trưng LBP.
PATCH_GRID = 4

#Cấu hình LBP đa tỉ lệ: (bán kính, số điểm lân cận).
LBP_CONFIGS = [
    (1, 8),
    (2, 16),
]

#Phương pháp tính LBP.
LBP_METHOD = "uniform"

#Cấu hình trích xuất đặc trưng HOG.
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}

#Các định dạng ảnh được chấp nhận.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def lbp_histogram(gray_patch, radius, points):
    """Tính biểu đồ histogram LBP cho một vùng ảnh xám."""
    lbp_image = local_binary_pattern(
        gray_patch,
        P=points,
        R=radius,
        method=LBP_METHOD,
    )

    #Với phương pháp uniform, số bin được sử dụng là points + 2.
    number_of_bins = points + 2

    histogram, _ = np.histogram(
        lbp_image.ravel(),
        bins=np.arange(number_of_bins + 1),
        range=(0, number_of_bins),
    )

    #Chuyển sang số thực và chuẩn hóa tổng histogram.
    histogram = histogram.astype(np.float32)
    histogram /= histogram.sum() + 1e-6

    return histogram


def extract_lbp_patch(gray_image):
    """Chia ảnh xám thành các patch và trích xuất LBP đa tỉ lệ."""
    height, width = gray_image.shape

    #Kích thước mỗi patch.
    patch_height = height // PATCH_GRID
    patch_width = width // PATCH_GRID

    if patch_height == 0 or patch_width == 0:
        raise ValueError("Ảnh quá nhỏ để chia thành các patch LBP.")

    features = []

    for row in range(PATCH_GRID):
        for col in range(PATCH_GRID):
            #Cắt từng patch từ ảnh gốc.
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

            #Tính LBP ở nhiều tỉ lệ cho từng patch.
            for radius, points in LBP_CONFIGS:
                histogram = lbp_histogram(
                    patch,
                    radius=radius,
                    points=points,
                )
                features.extend(histogram)

    return np.asarray(features, dtype=np.float32)


def extract_hog(gray_image):
    """Trích xuất đặc trưng HOG từ ảnh xám."""
    return hog(
        gray_image,
        orientations=HOG_PARAMS["orientations"],
        pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
        cells_per_block=HOG_PARAMS["cells_per_block"],
        block_norm=HOG_PARAMS["block_norm"],
        feature_vector=True,
    ).astype(np.float32)


def extract_features(image_path):
    """Đọc một ảnh, chuyển sang ảnh xám rồi ghép đặc trưng LBP và HOG."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    #Chuyển ảnh màu sang ảnh xám.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Chuẩn hóa kích thước ảnh về 128 x 128 để thống nhất HOG.
    gray_image = cv2.resize(
        gray_image,
        (128, 128),
        interpolation=cv2.INTER_AREA,
    )

    #Trích xuất từng loại đặc trưng.
    lbp_features = extract_lbp_patch(gray_image)
    hog_features = extract_hog(gray_image)

    #Ghép LBP và HOG thành một vector duy nhất.
    combined_features = np.concatenate(
        [lbp_features, hog_features]
    )

    return combined_features.astype(np.float32)


def get_image_paths(class_dir):
    """Lấy danh sách ảnh hợp lệ theo phần mở rộng tệp."""
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_dataset(split):
    """Đọc ảnh của một tập dữ liệu và tạo ma trận đặc trưng X, cùng vector nhãn y."""
    split_dir = DATA_DIR / split

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {split_dir}"
        )

    features_list = []
    labels_list = []

    print(f"\nĐang xử lý tập: {split.upper()}")

    for label, class_name in enumerate(CLASSES):
        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục lớp ảnh: {class_dir}"
            )

        image_paths = get_image_paths(class_dir)

        if not image_paths:
            print(
                f"  Cảnh báo: Lớp '{class_name}' không có ảnh "
                f"trong tập {split}."
            )
            continue

        print(
            f"  Lớp '{class_name}': "
            f"{len(image_paths)} ảnh"
        )

        successful_count = 0

        for index, image_path in enumerate(image_paths, start=1):
            try:
                features = extract_features(image_path)

                features_list.append(features)
                labels_list.append(label)
                successful_count += 1

            except (ValueError, cv2.error, OSError) as error:
                print(
                    f"    Bỏ qua ảnh lỗi: {image_path.name} "
                    f"({error})"
                )

        print(
            f"    Đã trích xuất thành công: "
            f"{successful_count}/{len(image_paths)} ảnh"
        )

    if not features_list:
        raise ValueError(
            f"Không trích xuất được đặc trưng nào cho tập '{split}'. "
            "Hãy kiểm tra dữ liệu đầu vào."
        )

    X = np.vstack(features_list).astype(np.float32)
    y = np.asarray(labels_list, dtype=np.int64)

    return X, y


def save_dataset(X_train, y_train, X_val, y_val, scaler):
    """Lưu dữ liệu đặc trưng, nhãn và bộ chuẩn hóa xuống ổ đĩa."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)

    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

    print("\nĐã lưu thành công các tệp:")
    print(f"  - {OUTPUT_DIR / 'X_train.npy'}")
    print(f"  - {OUTPUT_DIR / 'y_train.npy'}")
    print(f"  - {OUTPUT_DIR / 'X_val.npy'}")
    print(f"  - {OUTPUT_DIR / 'y_val.npy'}")
    print(f"  - {OUTPUT_DIR / 'scaler.pkl'}")


def main():
    """Thực hiện toàn bộ quy trình chuẩn bị dữ liệu cho SVM."""
    print("=" * 65)
    print("CHUẨN BỊ DỮ LIỆU VÀ TRÍCH XUẤT ĐẶC TRƯNG CHO SVM")
    print("=" * 65)

    print(f"\nThư mục dự án: {PROJECT_DIR}")
    print(f"Thư mục dữ liệu: {DATA_DIR}")
    print(f"Thư mục lưu kết quả: {OUTPUT_DIR}")
    print(f"Số lớp ảnh: {len(CLASSES)}")
    print(f"Lưới LBP: {PATCH_GRID} x {PATCH_GRID}")
    print(f"Cấu hình LBP: {LBP_CONFIGS}")
    print(f"Cấu hình HOG: {HOG_PARAMS}")

    print("\n[BƯỚC 1/3] TRÍCH XUẤT ĐẶC TRƯNG")

    X_train, y_train = load_dataset("train")
    X_val, y_val = load_dataset("val")

    print("\nKết quả trích xuất đặc trưng:")
    print(f"  - Train: X = {X_train.shape}, y = {y_train.shape}")
    print(f"  - Validation: X = {X_val.shape}, y = {y_val.shape}")

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            "Số chiều đặc trưng của train và validation không khớp."
        )

    print("\n[BƯỚC 2/3] CHUẨN HÓA ĐẶC TRƯNG")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    print("Đã chuẩn hóa đặc trưng thành công.")
    print(f"  - Kích thước X_train sau chuẩn hóa: {X_train.shape}")
    print(f"  - Kích thước X_val sau chuẩn hóa: {X_val.shape}")

    print("\n[BƯỚC 3/3] LƯU DỮ LIỆU")

    save_dataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        scaler=scaler,
    )

    print("\n" + "=" * 65)
    print("HOÀN TẤT CHUẨN BỊ DỮ LIỆU CHO SVM!")
    print("=" * 65)


if __name__ == "__main__":
    main()