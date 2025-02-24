from typing import Callable
import cv2 as cv
import torch
from config import BORDER_THICKNESS, COLOR, DEVICE, FONT, FONT_SCALE, MINIMAL_SCORE, VIDEO
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.transforms import transforms

weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
model.to(DEVICE)
model.eval()

preprocess = weights.transforms()

def start_video(video: str | int, draw_on_frame: Callable[[cv.typing.MatLike], cv.typing.MatLike]):
    cap = cv.VideoCapture(video)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 3)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to receive frame from camera")
            break

        cv.imshow("frame", draw_on_frame(frame))

        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
        
def predict(frame: cv.typing.MatLike):
    tensor = torch.from_numpy(frame).to(DEVICE).permute(2, 0, 1).float().div(255.0)

    return model([tensor])[0]

def draw_predictions(
        frame: cv.typing.MatLike, 
        minimal_score: float, 
        border_thickness: int,
        color: tuple[int, int, int],
        label_font: int,        
        font_scale: float
) -> cv.typing.MatLike:
    predictions = predict(frame)
    boxes, labels, scores = predictions["boxes"], predictions["labels"], predictions["scores"]
    
    for (box, label, score) in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        if score < minimal_score:
            continue

        [x1, x2] = map(int, box[::2])
        [y1, y2] = map(int, box[1::2])

        cv.rectangle(frame, (x1, y1), (x2, y2), color, border_thickness)
        cv.putText(frame, weights.meta["categories"][label], (x1, y1), label_font, font_scale, color, border_thickness)
       
    return frame

start_video(VIDEO, lambda frame: draw_predictions(frame, MINIMAL_SCORE, BORDER_THICKNESS, COLOR, FONT, FONT_SCALE))
