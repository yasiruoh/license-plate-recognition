"""Local video inference. Counts are per frame, never unique animal totals."""
import argparse
import json
from pathlib import Path
from src.engine import predict

def main():
    import cv2
    from PIL import Image
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Local video path")
    parser.add_argument("--output", default="outputs/frames.jsonl")
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    if args.stride < 1: parser.error("stride must be positive")
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): raise ValueError("Cannot open video")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("w", encoding="utf-8") as handle:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok: break
                if index % args.stride == 0:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result, _ = predict(image)
                    result["frame"] = index
                    handle.write(json.dumps(result) + "\n")
                index += 1
    finally:
        cap.release()

if __name__ == "__main__": main()
