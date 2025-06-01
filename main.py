import cv2
import numpy as np
import time



def main():
    model_path = "opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    input_size = (320, 320)  # you can use (320,320), (640,640), etc.—tradeoff speed vs. accuracy
    score_threshold = 0.5  # discard detections below 50% confidence
    nms_threshold = 0.3  # IoU threshold for NMS
    top_k = 1000  # keep up to 1000 highest-scoring boxes
    detector = cv2.FaceDetectorYN_create(
        model=model_path,
        config="",  # YuNet has no extra config file
        input_size=input_size,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k
    )

    # --- 3.2. Read and preprocess the image ---
    folder_name = "dev-images/"
    image_name = "green_cap_1916.jpg"
    image_path = folder_name+image_name
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}.jpg")

    # FaceDetectorYN expects BGR; you can pass the original or resize to speed up
    h, w = img.shape[:2]
    # If your image is much larger than input_size, you can downscale to approximately input_size
    # to speed up detection; but FaceDetectorYN will internally resize anyway.
    # For best results, pass the full-size image and let detector.resize internally.

    # Initialize detector’s expected input size (only needs to be done once,
    # but if your image shape != input_size, you should call setInputSize for each new shape)
    detector.setInputSize((w, h))  # set to the actual image size, NOT the model’s input_size

    # --- 3.3. Run detection ---
    # The output is a (N, 16) array: [x1, y1, x2, y2, score, ... five landmarks ...].
    # N = number of detected faces (≤ top_k)
    _, faces = detector.detect(img)

    # faces is None if no detections, otherwise a numpy array of shape (N, 16)
    if faces is None:
        print("No faces detected.")
        exit()

    # --- 3.4. Draw detections ---
    for i in range(faces.shape[0]):
        x1, y1, x2, y2, score = faces[i, :5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put confidence label
        label = f"{score:.2f}"
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- 3.5. Save or show result ---
    cv2.imwrite(f"results/{image_name}_yunet_detected.jpg", img)
    print(f"Detected {faces.shape[0]} face(s), saved to {image_name}_yunet_detected.jpg")

if __name__ == "__main__":
    main()