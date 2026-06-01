from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io
import os

from ultralytics import YOLO
import tensorflow as tf

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

yolo_model = YOLO(os.path.join(MODELS_DIR, "best_final.pt"))
contamination_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "contamination_model.h5"),
    compile=False
)

CLASS_NAMES = ["plastic", "glass", "paper", "metal", "non_recyclable", "batteries"]


def check_image_quality(image_bytes):
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(pil_image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness_score = gray.mean()

    result = {
        "is_blurry": blur_score < 80,
        "is_dark": brightness_score < 60,
        "is_too_bright": brightness_score > 210,
        "blur_score": float(blur_score),
        "brightness_score": float(brightness_score),
    }

    return result, pil_image


def enhance_image(pil_image, quality_result):
    enhanced = pil_image

    if quality_result["is_dark"]:
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.5)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)

    if quality_result["is_too_bright"]:
        enhanced = ImageEnhance.Brightness(enhanced).enhance(0.8)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.1)

    if quality_result["is_blurry"]:
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.8)

    return enhanced


def classify_contamination(cropped_bgr):
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(cropped_rgb, (224, 224))
    img_array = resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = contamination_model.predict(img_array, verbose=0)[0][0]

    print("Raw contamination prediction:", float(prediction))

    if prediction < 0.80:
        print("Final contamination label: Contaminated")
        return "Contaminated", float(1 - prediction)
    else:
        print("Final contamination label: Clean")
        return "Clean", float(prediction)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    quality_result, pil_image = check_image_quality(image_bytes)

    if quality_result["blur_score"] < 30:
        return jsonify({
            "status": "retake",
            "message": "The image is too blurry. Please retake the photo in better lighting and keep the camera steady."
        })

    if quality_result["brightness_score"] < 35:
        return jsonify({
            "status": "retake",
            "message": "The image is too dark. Please retake the photo with better lighting."
        })

    was_enhanced = False
    if (
        quality_result["is_blurry"] or
        quality_result["is_dark"] or
        quality_result["is_too_bright"]
    ):
        pil_image = enhance_image(pil_image, quality_result)
        was_enhanced = True

    image_np = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = yolo_model(image_bgr)

    detected_items = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cropped = image_bgr[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            material = CLASS_NAMES[cls_id]
            contamination, contamination_score = classify_contamination(cropped)

            recyclable = "Yes"
            if material in ["non_recyclable", "batteries"]:
                recyclable = "No"

            print("Material from YOLO:", material)
            print("Contamination from model:", contamination)
            print("Contamination score:", contamination_score)
            print("-----")

            detected_items.append({
                "material": material,
                "recyclable": recyclable,
                "contamination": contamination,
                "confidence": float(box.conf[0]),
                "contamination_confidence": contamination_score
            })

    if len(detected_items) == 0:
        return jsonify({
            "status": "retake",
            "message": "No waste item was detected. Please retake the image more clearly."
        })

    first_item = detected_items[0]

    return jsonify({
        "status": "success",
        "enhanced": was_enhanced,
        "material": first_item["material"],
        "recyclable": first_item["recyclable"],
        "contamination": first_item["contamination"],
        "contamination_confidence": first_item["contamination_confidence"],
        "all_detections": detected_items
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)