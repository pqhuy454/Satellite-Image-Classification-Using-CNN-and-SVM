from pathlib import Path

import shutil
import hashlib
import random
import cv2


#Xác định thư mục gốc của project
PROJECT_DIR = Path(__file__).resolve().parent.parent

#Đường dẫn dữ liệu
SOURCE_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "data_processed"

#Tên 5 lớp ảnh vệ tinh
CLASSES = ["forest", "lake", "residential", "river", "road"]

#Kích thước ảnh sau xử lý
IMG_SIZE = (128, 128)

# Tỷ lệ chia dữ liệu
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Seed để tái lập kết quả
SEED = 42


def compute_hash(image_path):
    """Tính mã băm MD5 để phát hiện ảnh trùng lặp."""
    hasher = hashlib.md5()

    with open(image_path, "rb") as file:
        hasher.update(file.read())

    return hasher.hexdigest()


def is_valid_image(image_path):
    """Kiểm tra ảnh có thể đọc được bằng OpenCV hay không."""
    img = cv2.imread(str(image_path))
    return img is not None


def prepare_output_folders():
    """Tạo thư mục train, val, test cho cả 5 lớp."""
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            path = OUTPUT_DIR / split / cls
            path.mkdir(parents=True, exist_ok=True)


def resize_and_save(src_path, dst_path):
    """Resize ảnh về kích thước quy định và lưu vào thư mục đích."""
    img = cv2.imread(str(src_path))

    if img is None:
        return False

    resized_img = cv2.resize(img, IMG_SIZE)

    return cv2.imwrite(str(dst_path), resized_img)


def clean_and_split_class(class_name):
    """Làm sạch, lọc ảnh trùng, resize và chia dữ liệu cho một lớp."""
    src_dir = SOURCE_DIR / class_name

    if not src_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu của lớp: {src_dir}"
        )

    images = []
    seen_hashes = set()

    removed_duplication = 0
    removed_error = 0

    #Kiểm tra ảnh lỗi và lọc ảnh trùng
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue

        if not is_valid_image(img_path):
            removed_error += 1
            continue

        img_hash = compute_hash(img_path)

        if img_hash in seen_hashes:
            removed_duplication += 1
            continue

        seen_hashes.add(img_hash)
        images.append(img_path)

    #Xáo trộn dữ liệu trước khi chia
    random.shuffle(images)

    total_images = len(images)

    #Chia dữ liệu theo tỷ lệ 70/15/15
    train_size = int(total_images * TRAIN_RATIO)
    val_size = int(total_images * VAL_RATIO)

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    #Resize và lưu ảnh vào từng tập
    for split_name, split_images in splits.items():
        for image_path in split_images:
            dst_path = (
                OUTPUT_DIR
                / split_name
                / class_name
                / image_path.name
            )

            if not resize_and_save(image_path, dst_path):
                print(f"Cảnh báo: Không thể lưu ảnh {image_path}")

    print(
        f"{class_name}: tổng ảnh hợp lệ = {total_images} | "
        f"train = {len(train_images)} | "
        f"val = {len(val_images)} | "
        f"test = {len(test_images)} | "
        f"ảnh trùng = {removed_duplication} | "
        f"ảnh lỗi = {removed_error}"
    )


def main():
    #Kiểm tra dữ liệu nguồn
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu nguồn: {SOURCE_DIR}"
        )

    #Kiểm tra đủ 5 lớp dữ liệu
    for cls in CLASSES:
        class_dir = SOURCE_DIR / cls

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục lớp '{cls}': {class_dir}"
            )

    #Kiểm tra tổng tỷ lệ chia dữ liệu
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError(
            "Tổng tỷ lệ train/val/test phải bằng 1."
        )

    #Thiết lập seed để tái lập kết quả
    random.seed(SEED)

    print("=" * 60)
    print("BẮT ĐẦU LÀM SẠCH VÀ CHIA DỮ LIỆU ẢNH VỆ TINH")
    print(f"Dữ liệu nguồn : {SOURCE_DIR}")
    print(f"Dữ liệu đầu ra: {OUTPUT_DIR}")
    print("=" * 60)

    #Xóa dữ liệu đầu ra cũ để tránh dữ liệu tồn dư
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    #Tạo cấu trúc thư mục đầu ra
    prepare_output_folders()

    #Xử lý lần lượt cả 5 lớp
    for cls in CLASSES:
        clean_and_split_class(cls)

    print("\n" + "=" * 60)
    print("HOÀN TẤT LÀM SẠCH VÀ CHIA DỮ LIỆU!")
    print(f"Dữ liệu đã lưu tại: {OUTPUT_DIR}")
    print("Các tập dữ liệu: train / val / test")
    print("=" * 60)


if __name__ == "__main__":
    main()