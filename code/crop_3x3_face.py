from pathlib import Path
import csv
import cv2
import numpy as np


RAW_DIR = Path("rubik_dataset")
ROI_DIR = Path("rubik_dataset/roi")
METADATA_PATH = Path("rubik_dataset/metadata_roi.csv")

LABELS = ["putih", "kuning", "merah", "oranye", "biru", "hijau"]

WARP_SIZE = 600
GRID_SIZE = 3
CELL_SIZE = WARP_SIZE // GRID_SIZE

# Ambil bagian tengah cell agar border hitam tidak terlalu masuk.
CENTER_CROP_RATIO = 0.80


clicked_points = []


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Urutkan 4 titik menjadi:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]       # top-left
    rect[2] = points[np.argmax(s)]       # bottom-right
    rect[1] = points[np.argmin(diff)]    # top-right
    rect[3] = points[np.argmax(diff)]    # bottom-left

    return rect


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points)

    dst = np.array(
        [
            [0, 0],
            [WARP_SIZE - 1, 0],
            [WARP_SIZE - 1, WARP_SIZE - 1],
            [0, WARP_SIZE - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (WARP_SIZE, WARP_SIZE))

    return warped


def center_crop_cell(cell: np.ndarray, ratio: float) -> np.ndarray:
    h, w = cell.shape[:2]

    new_h = int(h * ratio)
    new_w = int(w * ratio)

    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2

    return cell[y1:y1 + new_h, x1:x1 + new_w]


def mouse_callback(event, x, y, flags, param):
    global clicked_points

    display_image = param["display_image"]
    scale = param["scale"]

    if event == cv2.EVENT_LBUTTONDOWN:
        original_x = int(x / scale)
        original_y = int(y / scale)

        clicked_points.append([original_x, original_y])

        cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(
            display_image,
            str(len(clicked_points)),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow(param["window_name"], display_image)


def resize_for_display(image: np.ndarray, max_width: int = 900):
    h, w = image.shape[:2]

    if w <= max_width:
        return image.copy(), 1.0

    scale = max_width / w
    new_h = int(h * scale)
    resized = cv2.resize(image, (max_width, new_h))

    return resized, scale


def crop_image(image_path: Path, label: str, writer):
    global clicked_points
    clicked_points = []

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[SKIP] Failed to read: {image_path}")
        return

    display_image, scale = resize_for_display(image)

    window_name = f"Click 4 corners: {image_path.name}"
    cv2.namedWindow(window_name)

    callback_param = {
        "display_image": display_image.copy(),
        "scale": scale,
        "window_name": window_name,
    }

    cv2.setMouseCallback(window_name, mouse_callback, callback_param)

    print()
    print(f"Image: {image_path}")
    print("Klik 4 sudut luar sisi Rubik.")
    print("Urutan bebas, nanti akan diurutkan otomatis.")
    print("Tekan:")
    print("  ENTER = proses crop")
    print("  r     = reset titik")
    print("  s     = skip gambar")
    print("  q     = keluar")

    while True:
        cv2.imshow(window_name, callback_param["display_image"])
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            if len(clicked_points) != 4:
                print("[WARN] Harus klik tepat 4 titik.")
                continue
            break

        if key == ord("r"):
            clicked_points = []
            callback_param["display_image"] = display_image.copy()
            print("[RESET] Titik dihapus. Klik ulang 4 sudut.")

        if key == ord("s"):
            print(f"[SKIP] {image_path.name}")
            cv2.destroyWindow(window_name)
            return

        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit

    cv2.destroyWindow(window_name)

    points = np.array(clicked_points, dtype="float32")
    warped = four_point_transform(image, points)

    label_dir = ROI_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    image_id = image_path.stem

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y1 = row * CELL_SIZE
            y2 = (row + 1) * CELL_SIZE
            x1 = col * CELL_SIZE
            x2 = (col + 1) * CELL_SIZE

            cell = warped[y1:y2, x1:x2]
            roi = center_crop_cell(cell, CENTER_CROP_RATIO)

            sample_id = f"{image_id}_r{row}_c{col}"
            roi_path = label_dir / f"{sample_id}.jpg"

            cv2.imwrite(str(roi_path), roi)

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "row": row,
                    "col": col,
                    "label": label,
                    "source_path": str(image_path),
                    "roi_path": str(roi_path),
                    "is_augmented": False,
                    "source_id": sample_id,
                }
            )

    print(f"[OK] Saved 9 ROI from {image_path.name}")


def main():
    ROI_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]

    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "sample_id",
            "image_id",
            "row",
            "col",
            "label",
            "source_path",
            "roi_path",
            "is_augmented",
            "source_id",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for label in LABELS:
            label_dir = RAW_DIR / label

            if not label_dir.exists():
                print(f"[WARN] Folder not found: {label_dir}")
                continue

            image_paths = []

            for ext in image_extensions:
                image_paths.extend(label_dir.glob(ext))

            image_paths = sorted(image_paths)

            print(f"\nLabel: {label}")
            print(f"Found {len(image_paths)} images.")

            for image_path in image_paths:
                crop_image(image_path, label, writer)

    cv2.destroyAllWindows()
    print(f"\nDone. Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()