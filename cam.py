import argparse
import sys
import time
from functools import lru_cache
import threading
from collections import Counter, deque
import subprocess
from datetime import datetime
import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

class DetectionStats:
    def __init__(self, window_size=30):  # Reduced window size
        self.window_size = window_size
        self.recent_detections = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.label_counts = Counter()
        self.total_frames = 0
        self.frames_with_detections = 0
        self.lock = threading.Lock()
        self.detection_history = deque(maxlen=5)  # Reduced history size
        
    def update(self, detections, labels):
        with self.lock:
            self.total_frames += 1
            if detections:
                self.frames_with_detections += 1
                self.recent_detections.append(len(detections))
                
                for det in detections:
                    self.confidence_history.append(det.conf)
                    self.label_counts[det.category] += 1
                    
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
            
            label_stats = {labels[int(cat)]: count 
                          for cat, count in self.label_counts.most_common(3)}
            
            return {
                "average_detections": round(avg_detections, 1),
                "average_confidence": round(avg_confidence * 100, 1),
                "detection_rate": round((self.frames_with_detections / self.total_frames) * 100, 1),
                "common_objects": label_stats,
                "recent_history": list(self.detection_history)
            }

class StreamBuffer:
    def __init__(self, max_size=2):
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def put(self, frame):
        with self.lock:
            self.queue.append(frame)
            
    def get(self):
        with self.lock:
            return self.queue.popleft() if self.queue else None

class TwitchStreamer(threading.Thread):
    def __init__(self, rtmp_url, stream_key, width=640, height=480, fps=15):
        super().__init__()
        self.stream_url = f"{rtmp_url}/{stream_key}"
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer = StreamBuffer()
        self.running = True
        self.process = None
        
    def run(self):
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{self.width}x{self.height}",
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-f', 'flv',
            '-flvflags', 'no_duration_filesize',
            '-bufsize', '512k',
            self.stream_url
        ]
        
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        
        while self.running:
            frame = self.buffer.get()
            if frame is not None:
                try:
                    self.process.stdin.write(frame.tobytes())
                except (IOError, BrokenPipeError):
                    break
                    
        self.stop()
    
    def send_frame(self, frame):
        self.buffer.put(frame)
    
    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                self.process.kill()
            self.process = None

class Detection:
    __slots__ = ['category', 'conf', 'box']
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

@lru_cache(maxsize=32)
def get_labels():
    labels = intrinsics.labels
    return [label for label in labels if label and label != "-"] if intrinsics.ignore_dash_labels else labels

def draw_overlay(frame, stats, fps):
    # Create overlay only when needed
    if stats:
        height, width = frame.shape[:2]
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Simplified stats display
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y = 50
        for key, value in stats.items():
            if key in ['average_detections', 'average_confidence', 'detection_rate']:
                cv2.putText(overlay, f"{key}: {value}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 20
        
        # Blend overlay
        mask = overlay > 0
        frame[mask] = cv2.addWeighted(frame[mask], 0.5, overlay[mask], 0.5, 0)
    
    return frame

def main(args):
    global intrinsics, imx500, picam2
    
    detection_stats = DetectionStats()
    labels = get_labels()
    
    # Initialize and start streamer in separate thread
    streamer = TwitchStreamer(
        args.rtmp_url,
        args.stream_key,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    streamer.start()
    
    fps_tracker = deque(maxlen=30)
    last_time = time.time()
    
    try:
        while True:
            frame_start = time.time()
            
            metadata = picam2.capture_metadata()
            frame = picam2.capture_array()
            
            # Process detections only every other frame
            if streamer.buffer.queue.maxlen > len(streamer.buffer.queue):
                detections = parse_detections(metadata)
                if detections:
                    detection_stats.update(detections, labels)
                    frame = draw_detections(frame, detections, labels)
            
                # Calculate FPS
                current_time = time.time()
                fps_tracker.append(1 / (current_time - frame_start))
                fps = sum(fps_tracker) / len(fps_tracker)
                
                # Update overlay
                stats = detection_stats.get_stats(labels)
                frame = draw_overlay(frame, stats, fps)
            
            # Send frame to streamer
            streamer.send_frame(frame)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        streamer.stop()
        streamer.join()
        picam2.stop()
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=5)
    parser.add_argument("--rtmp-url", type=str, required=True)
    parser.add_argument("--stream-key", type=str, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    
    args = parser.parse_args()

    # Initialize camera with optimized settings
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    if not intrinsics.task:
        intrinsics.task = "object detection"
    
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (args.width, args.height)},
        controls={"FrameRate": args.fps},
        buffer_count=4  # Reduced buffer count
    )
    
    picam2.start(config)
    main(args)
