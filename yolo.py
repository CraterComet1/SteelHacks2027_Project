from ultralytics import YOLO

model = YOLO("yolo26n.pt")
confidence_threshold = 0.5

def detect(image):

    results = model(
        image,
        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            if confidence < confidence_threshold:
                continue

            class_id = int(box.cls[0])

            class_name = model.names[
                class_id
            ]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            detections.append({
                "class": class_name,
                "class_id": class_id,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })

    return detections