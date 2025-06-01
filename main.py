import cv2
import numpy as np
import os



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
        filename = filename.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").replace(".bmp", "")
        if not os.path.exists(f"results/{run_number}"):
            os.makedirs(f"results/{run_number}")
        cv2.imwrite(f"results/{run_number}/{filename}_yunet_detected_{input_size_nr}.jpg", img)
        print(f"Detected {faces.shape[0]} face(s), saved to results/{run_number}/{filename}_yunet_detected_{input_size_nr}.jpg")

if __name__ == "__main__":
    folder_name = "dev-images/"
    model_path = "opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    input_size_nr_list = [320, 480, 640, 960, 1280, 1600, 1920, 2560]  # Different input sizes to test
    score_threshold_list = [i/10 for i in range(1, 10)]  # Score thresholds from 0 to 100 in steps of 10
    nms_threshold_list = [i/10 for i in range(1, 10)]
    top_k_list = list(range(100, 1000, 100))
    run_number = 1
    for input_size_nr in input_size_nr_list:
        for score_threshold in score_threshold_list:
            for nms_threshold in nms_threshold_list:
                for top_k in top_k_list:
                    print(f"Running with input_size={input_size_nr}, score_threshold={score_threshold}, nms_threshold={nms_threshold}")
                    main(folder_name=folder_name, model_path=model_path, input_size_nr=input_size_nr, run_number=run_number, score_threshold=score_threshold, nms_threshold=nms_threshold, top_k=top_k)