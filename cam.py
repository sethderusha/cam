import argparse
import sys
import time
from functools import lru_cache
import threading
from collections import Counter, deque
import subprocess
from datetime import datetime
import os
import csv
from queue import Queue

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# Statistics tracking
class DetectionStats:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.recent_detections = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.label_counts = Counter()
        self.total_frames = 0
        self.frames_with_detections = 0
        self.lock = threading.Lock()
        
        # Keep track of recent detections for display
        self.detection_history = deque(maxlen=10)  # Store last 10 detections
        
    def update(self, detections, labels):
        with self.lock:
            self.total_frames += 1
            if detections:
                self.frames_with_detections += 1
                self.recent_detections.append(len(detections))
                
                # Update confidence history and label counts
                for det in detections:
                    self.confidence_history.append(det.conf)
                    self.label_counts[det.category] += 1
                    
                    # Add to detection history with label
                    self.detection_history.append({
                        'label': labels[int(det.category)],
                        'confidence': det.conf,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
                
    def get_stats(self, labels):
        with self.lock:
            if not self.recent_detections:
                return {}
                
            avg_detections = sum(self.recent_detections) / len(self.recent_detections)
            avg_confidence = sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else 0
            
            # Convert category indices to label names
            label_stats = {labels[int(cat)]: count 
                          for cat, count in self.label_counts.most_common(5)}
            
            return {
                "average_detections": round(avg_detections, 2),
                "average_confidence": round(avg_confidence * 100, 2),
                "detection_rate": round((self.frames_with_detections / self.total_frames) * 100, 2),
                "common_objects": label_stats,
                "total_frames": self.total_frames,
                "recent_history": list(self.detection_history)
            }

class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

class TwitchStreamer:
    def __init__(self, rtmp_url, stream_key, width=1280, height=720, fps=30):
        self.rtmp_url = rtmp_url
        self.stream_key = stream_key
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        
    def start_stream(self):
        stream_url = f"{self.rtmp_url}/{self.stream_key}"
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24', #Important to match your frame format
            '-s', f"{args.width}x{args.height}", #Make sure this matches the config
            '-r', str(args.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'flv',
            stream_url
        ]
        
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        
    def send_frame(self, frame):
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(frame.tobytes())
            except IOError as e:
                print(f"Error writing to FFmpeg: {e}")
                self.stop_stream()
                
    def stop_stream(self):
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            self.process = None

def parse_detections(metadata: dict):
    """Parse the output tensor into detections."""
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return None
        
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    return [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]

@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def draw_overlay(frame, stats, fps):
    """Draw statistics overlay on the frame."""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay for stats background
    overlay = frame.copy()
    
    # Draw stats panel background
    cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw stats
    y = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y), font, 0.6, (255, 255, 255), 1)
    y += 25
    
    # Detection stats - Use .get() with default values
    avg_detections = stats.get('average_detections', 0.0)
    avg_confidence = stats.get('average_confidence', 0.0)
    detection_rate = stats.get('detection_rate', 0.0)

    cv2.putText(frame, f"Avg Detections: {avg_detections:.1f}", (20, y), font, 0.6, (255, 255, 255), 1)
    y += 25
    cv2.putText(frame, f"Confidence: {avg_confidence:.1f}%", (20, y), font, 0.6, (255, 255, 255), 1)
    y += 25
    cv2.putText(frame, f"Detection Rate: {detection_rate:.1f}%", (20, y), font, 0.6, (255, 255, 255), 1)
    y += 25
    
    # Common objects
    common_objects = stats.get('common_objects', {})
    cv2.putText(frame, "Most Common Objects:", (20, y), font, 0.6, (255, 255, 255), 1)
    y += 25
    for label, count in list(common_objects.items())[:3]: #list cast prevents runtime errors
        cv2.putText(frame, f"- {label}: {count}", (30, y), font, 0.6, (255, 255, 255), 1)
        y += 25

    # Recent detections panel
    recent_history = stats.get('recent_history', [])
    if recent_history:  # Only draw if there's history
        # Draw background for recent detections
        cv2.rectangle(overlay, (width - 310, 10), (width - 10, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "Recent Detections:", (width - 300, 40), font, 0.6, (255, 255, 255), 1)
        y = 70
        for det in reversed(recent_history):
            text = f"[{det['timestamp']}] {det['label']} ({det['confidence']:.2f})"
            cv2.putText(frame, text, (width - 300, y), font, 0.5, (255, 255, 255), 1)
            y += 25
            if y > 230:  # Limit the number of displayed detections
                break
    
    return frame

def draw_detections(frame, detections, labels):
    """Draw detections onto the frame."""
    for detection in detections:
        x, y, w, h = detection.box
        label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

        # Draw detection box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - text_h - 4), (x + text_w, y), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame

def main(args):
    global intrinsics, imx500, picam2
    
    # Initialize detection components
    detection_stats = DetectionStats()
    labels = get_labels()
    
    # Initialize Twitch streamer
    streamer = TwitchStreamer(
        args.rtmp_url,
        args.stream_key,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    
    # Start streaming
    streamer.start_stream()
    
    # FPS tracking
    fps = 0
    frame_count = 0
    last_time = time.time()
    
    try:
        while True:
            # Capture and process frame
            metadata = picam2.capture_metadata()
            frame = picam2.capture_array()
            
            # Update FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_time > 1.0:
                fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
            
            # Get detections
            detections = parse_detections(metadata)
            
            if detections:
                # Update statistics
                detection_stats.update(detections, labels)
                
                # Draw detections on frame
                frame = draw_detections(frame, detections, labels)
            
            # Get current stats
            stats = detection_stats.get_stats(labels)
            
            # Draw overlay with stats and recent detections
            frame = draw_overlay(frame, stats, fps)
            
            # Stream frame
            streamer.send_frame(np.ascontiguousarray(frame))
            
    except KeyboardInterrupt:
        print("Shutting down...")
        streamer.stop_stream()
        picam2.stop()
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add existing arguments
    parser.add_argument("--model", type=str,
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--max-detections", type=int, default=10)
    
    # Add Twitch-specific arguments
    parser.add_argument("--rtmp-url", type=str, required=True,
                        help="Twitch RTMP URL (e.g., rtmp://live.twitch.tv/app)")
    parser.add_argument("--stream-key", type=str, required=True,
                        help="Twitch stream key")
    parser.add_argument("--width", type=int, default=1280,
                        help="Output stream width")
    parser.add_argument("--height", type=int, default=720,
                        help="Output stream height")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output stream FPS")
    
    args = parser.parse_args()

    # Initialize camera and detection components
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (args.width, args.height)}, #Force RGB888 and size
        controls={"FrameRate": args.fps},
        buffer_count=12
    )
    
    picam2.start(config)
    
    main(args)
