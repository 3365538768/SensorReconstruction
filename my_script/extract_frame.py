import os
import cv2
import argparse

def extract_frames(input_path, output_dir, target_fps):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(orig_fps / target_fps))
    if frame_interval <= 0:
        frame_interval = 1

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    digit_count = len(str(frame_count))
    
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at the specified interval
        if frame_idx % frame_interval == 0:
            filename = f"frame_{saved_count:0{digit_count}d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video at a specified FPS.")
    parser.add_argument("input_path", type=str, help="Path to the input .mov video file")
    parser.add_argument("output_dir", type=str, help="Directory to save extracted frames")
    parser.add_argument("fps", type=float, help="Target frames per second to extract")

    args = parser.parse_args()
    extract_frames(args.input_path, args.output_dir, args.fps)
