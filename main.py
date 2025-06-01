import cv2
import numpy as np
import os
from mtcnn import MTCNN
import numpy as np

# ───── 2.3. Helper: resize with aspect ratio ────────────────────────────────────────────────────
def resize_with_aspect(img, max_side):
    """
    Downscale img so that its longest side == max_side (if it’s bigger).
    Returns (resized_img, scale_x, scale_y).
    If img is already smaller, returns (img, 1.0, 1.0).
    """
    h, w = img.shape[:2]
    # If the image is already <= max_side in both dimensions, do nothing.
    if max(h, w) <= max_side:
        return img, 1.0, 1.0

    # Compute scale so that max(h,w)*scale = max_side
    if w >= h:
        scale = max_side / w
    else:
        scale = max_side / h

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    return resized, scale, scale

def main(folder_name, model_path, input_size_nr, run_number, score_threshold, nms_threshold, top_k):
    input_size = (input_size_nr, input_size_nr)  # you can use (320,320), (640,640), etc.—tradeoff speed vs. accuracy
    score_threshold = score_threshold # discard detections below X confidence
    nms_threshold = nms_threshold  # IoU threshold for NMS
    top_k = top_k  # keep up to 1000 highest-scoring boxes
    detector = cv2.FaceDetectorYN_create(
        model=model_path,
        config="",  # YuNet has no extra config file
        input_size=input_size,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k
    )

    # --- 3.2. Read and preprocess the image ---
    for filename in os.listdir(folder_name):
        # skip non-image files (adjust extensions as needed)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        image_path = folder_name+filename
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise FileNotFoundError(f"Could not load {image_path}.jpg")

        # 1) Downscale so the longest side is MAX_SIDE_FOR_DETECTION px
        MAX_SIDE_FOR_DETECTION = 4320
        small_img, sx, sy = resize_with_aspect(img=orig_img, max_side=MAX_SIDE_FOR_DETECTION)

        # FaceDetectorYN expects BGR; you can pass the original or resize to speed up
        h, w = small_img.shape[:2]
        # If your image is much larger than input_size, you can downscale to approximately input_size
        # to speed up detection; but FaceDetectorYN will internally resize anyway.
        # For best results, pass the full-size image and let detector.resize internally.

        # Initialize detector’s expected input size (only needs to be done once,
        # but if your image shape != input_size, you should call setInputSize for each new shape)
        detector.setInputSize((w, h))  # set to the actual image size, NOT the model’s input_size

        # --- 3.3. Run detection ---
        # The output is a (N, 16) array: [x1, y1, x2, y2, score, ... five landmarks ...].
        # N = number of detected faces (≤ top_k)
        _, faces = detector.detect(small_img)

        # faces is None if no detections, otherwise a numpy array of shape (N, 16)
        if faces is None:
            print("No faces detected.")
            continue

        # --- 3.4. Draw detections ---
        for i in range(faces.shape[0]):
            x1, y1, x2, y2, score = faces[i, :5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Draw bounding box
            cv2.rectangle(small_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put confidence label
            label = f"{score:.2f}"
            cv2.putText(small_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- 3.5. Save or show result ---
        filename = filename.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").replace(".bmp", "")
        if not os.path.exists(f"results/{run_number}"):
            os.makedirs(f"results/{run_number}")
        cv2.imwrite(f"results/{run_number}/{filename}_yunet_detected_input_{input_size_nr}_score_{score_threshold}_nms_{nms_threshold}_top_{top_k}.jpg", small_img)
        #print(f"Detected {faces.shape[0]} face(s), saved to results/{run_number}/{filename}_yunet_detected_input_{input_size_nr}_score_{score_threshold}_nms_{nms_threshold}_top_{top_k}.jpg")



def mtcnn_detected_faces(folder_name, run_number):
    # This function is a placeholder for MTCNN detection logic.
    # You can implement MTCNN detection here if needed.
    mtcnn = MTCNN()

    for filename in os.listdir(folder_name):
        # skip non-image files (adjust extensions as needed)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        image_path = folder_name + filename
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise FileNotFoundError(f"Could not load {image_path}.jpg")

        MAX_SIDE_FOR_DETECTION = 1280
        small, sx, sy = resize_with_aspect(img=orig_img, max_side=MAX_SIDE_FOR_DETECTION)
        img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        detections = mtcnn.detect_faces(img_rgb)

        if not detections:
            print(f"→ MTCNN: no faces in {filename}")
            continue

        for det in detections:
            x, y, w, h = det["box"]
            score = det["confidence"]
            if score < 0.8:  # tune as needed
                continue

            # Rescale
            x1o = int(x)
            y1o = int(y)
            x2o = int((x + w))
            y2o = int((y + h))

            cv2.rectangle(small, (x1o, y1o), (x2o, y2o), (255, 0, 0), 2)
            cv2.putText(
                small,
                f"{score:.2f}",
                (x1o, y1o - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # --- 3.5. Save or show result ---
        filename = filename.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").replace(".bmp", "")
        if not os.path.exists(f"results/{run_number}"):
            os.makedirs(f"results/{run_number}")
        cv2.imwrite(
            f"results/{run_number}/{filename}_MTCNN.jpg",
            small)
        # print(f"Detected {faces.shape[0]} face(s), saved to results/{run_number}/{filename}_yunet_detected_input_{input_size_nr}_score_{score_threshold}_nms_{nms_threshold}_top_{top_k}.jpg")


if __name__ == "__main__":
    folder_name = "dev-images/"

    run_number = 13
    mtcnn_detected_faces(folder_name=folder_name, run_number=run_number)

    """
    model_path = "opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    achor_YUNET_input_size_nr_list = [320, 480, 640, 960, 1280]  # Different input sizes to test
    score_threshold_list = [i/10 for i in range(3, 10)]  # Score thresholds from 0 to 100 in steps of 10
    nms_threshold_list = [i/10 for i in range(3, 10)]
    top_k = 100
    test_number_iter = 11
    run_number = 1
    total_runs = len(achor_YUNET_input_size_nr_list) * len(score_threshold_list) * len(nms_threshold_list)
    # 2) How large (in px) the **longest** side of the input image we feed into YuNet.
    #    This trades off speed vs. small-face recall. 1280 is a good midpoint for a 6000×4000.
    MAX_SIDE_FOR_DETECTION = 1280
    for anchor_input_size_nr in achor_YUNET_input_size_nr_list:
        for score_threshold in score_threshold_list:
            for nms_threshold in nms_threshold_list:
                print(f"Running with input_size={anchor_input_size_nr}, score_threshold={score_threshold}, nms_threshold={nms_threshold} || Run number/total: {run_number}/{total_runs}")
                main(folder_name=folder_name, model_path=model_path, input_size_nr=anchor_input_size_nr, run_number=test_number_iter, score_threshold=score_threshold, nms_threshold=nms_threshold, top_k=top_k)
                run_number = run_number + 1
                """