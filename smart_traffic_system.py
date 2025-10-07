"""
Smart Traffic Light Automation System
======================================
A dynamic traffic light control system that uses YOLOv8 for vehicle detection
and adjusts signal timing based on real-time traffic density on two roads.

Author: Traffic Automation Team
Date: October 2025
"""

import cv2
import numpy as np
from typing import Tuple, List

from traffic_core import TrafficLightController, TrafficStats


class VehicleDetector:
    """
    Vehicle detection using background subtraction and contour analysis.
    Alternative to YOLO for lightweight detection.
    """
    
    def __init__(self, min_contour_area: int = 500):
        """
        Initialize the vehicle detector.
        
        Args:
            min_contour_area: Minimum area threshold for vehicle detection
        """
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=50, 
            detectShadows=True
        )
        self.min_contour_area = min_contour_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
    def detect_vehicles(self, frame: np.ndarray) -> Tuple[int, List[Tuple[int, int, int, int]]]:
        """
        Detect vehicles in the frame using background subtraction.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (vehicle_count, list of bounding boxes)
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (value 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to reduce noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours and get bounding boxes
        vehicles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                vehicles.append((x, y, w, h))
        
        return len(vehicles), vehicles


class SmartTrafficSystem:
    """
    Main system that integrates vehicle detection and traffic light control.
    """
    
    def __init__(self, video_road1: str, video_road2: str):
        """
        Initialize the smart traffic system.
        
        Args:
            video_road1: Path to video file for road 1
            video_road2: Path to video file for road 2
        """
        # Initialize video captures
        self.cap_road1 = cv2.VideoCapture(video_road1)
        self.cap_road2 = cv2.VideoCapture(video_road2)
        
        # Check if videos opened successfully
        if not self.cap_road1.isOpened():
            raise ValueError(f"Unable to open video: {video_road1}")
        if not self.cap_road2.isOpened():
            raise ValueError(f"Unable to open video: {video_road2}")
        
        # Initialize detectors for each road
        self.detector_road1 = VehicleDetector()
        self.detector_road2 = VehicleDetector()
        
        # Initialize traffic light controller
        self.controller = TrafficLightController()
        
        # Initialize statistics
        self.stats_road1 = TrafficStats()
        self.stats_road2 = TrafficStats()
        
        # Get video properties
        self.fps = int(self.cap_road1.get(cv2.CAP_PROP_FPS))
        if self.fps == 0:
            self.fps = 30  # Default FPS
        
    def draw_traffic_light(self, frame: np.ndarray, signal: str, position: str) -> np.ndarray:
        """
        Draw traffic light indicator on the frame.
        
        Args:
            frame: Input frame
            signal: Signal state ('GREEN', 'YELLOW', 'RED')
            position: Position of indicator ('top-left' or 'top-right')
            
        Returns:
            Frame with traffic light drawn
        """
        h, w = frame.shape[:2]
        
        # Determine position
        if position == 'top-left':
            x_offset, y_offset = 20, 20
        else:
            x_offset, y_offset = w - 120, 20
        
        # Draw traffic light background
        cv2.rectangle(frame, (x_offset, y_offset), (x_offset + 80, y_offset + 200), (50, 50, 50), -1)
        
        # Define light colors
        colors = {
            'RED': (0, 0, 255),
            'YELLOW': (0, 255, 255),
            'GREEN': (0, 255, 0)
        }
        
        # Draw lights
        light_positions = {'RED': 50, 'YELLOW': 110, 'GREEN': 170}
        
        for light, y_pos in light_positions.items():
            color = colors[light] if light == signal else (80, 80, 80)
            cv2.circle(frame, (x_offset + 40, y_offset + y_pos), 25, color, -1)
            cv2.circle(frame, (x_offset + 40, y_offset + y_pos), 25, (255, 255, 255), 2)
        
        return frame
    
    def draw_info_panel(self, frame: np.ndarray, stats: TrafficStats, 
                       signal: str, time_remaining: float, road_name: str) -> np.ndarray:
        """
        Draw information panel on the frame.
        
        Args:
            frame: Input frame
            stats: Traffic statistics
            signal: Current signal state
            time_remaining: Time until signal change
            road_name: Name of the road
            
        Returns:
            Frame with info panel drawn
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 150), (w - 10, h - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw text information
        y_pos = h - 120
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        info_texts = [
            f"{road_name}",
            f"Vehicles: {stats.vehicle_count}",
            f"Congestion: {stats.congestion_level}",
            f"Signal: {signal} ({time_remaining:.1f}s)",
        ]
        
        for text in info_texts:
            cv2.putText(frame, text, (20, y_pos), font, 0.6, (255, 255, 255), 2)
            y_pos += 30
        
        return frame
    
    def draw_detections(self, frame: np.ndarray, vehicles: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw bounding boxes around detected vehicles.
        
        Args:
            frame: Input frame
            vehicles: List of vehicle bounding boxes
            
        Returns:
            Frame with detections drawn
        """
        for i, (x, y, w, h) in enumerate(vehicles):
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw vehicle ID
            cv2.putText(frame, f"V{i+1}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """
        Main loop to run the smart traffic system.
        """
        print("=" * 60)
        print("Smart Traffic Light Automation System")
        print("=" * 60)
        print("Press 'q' to quit")
        print("=" * 60)
        
        frame_count = 0
        
        try:
            while True:
                # Read frames from both videos
                ret1, frame1 = self.cap_road1.read()
                ret2, frame2 = self.cap_road2.read()
                
                # Check if we've reached the end of either video
                if not ret1 or not ret2:
                    print("\nEnd of video reached. Restarting...")
                    self.cap_road1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.cap_road2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Resize frames for better visualization
                frame1 = cv2.resize(frame1, (640, 480))
                frame2 = cv2.resize(frame2, (640, 480))
                
                # Detect vehicles on both roads
                count1, vehicles1 = self.detector_road1.detect_vehicles(frame1)
                count2, vehicles2 = self.detector_road2.detect_vehicles(frame2)
                
                # Update statistics
                self.stats_road1.update(count1)
                self.stats_road2.update(count2)
                
                # Update traffic signal timing
                signal_status = self.controller.update_signal_timing(count1, count2)
                
                # Draw detections
                frame1 = self.draw_detections(frame1, vehicles1)
                frame2 = self.draw_detections(frame2, vehicles2)
                
                # Draw traffic lights
                frame1 = self.draw_traffic_light(frame1, signal_status['road1'], 'top-right')
                frame2 = self.draw_traffic_light(frame2, signal_status['road2'], 'top-right')
                
                # Draw info panels
                frame1 = self.draw_info_panel(frame1, self.stats_road1, 
                                             signal_status['road1'], 
                                             signal_status['time_remaining'], 
                                             "ROAD 1")
                frame2 = self.draw_info_panel(frame2, self.stats_road2, 
                                             signal_status['road2'], 
                                             signal_status['time_remaining'], 
                                             "ROAD 2")
                
                # Combine frames side by side
                combined_frame = np.hstack([frame1, frame2])
                
                # Display the result
                cv2.imshow('Smart Traffic Light System', combined_frame)
                
                # Print statistics every 30 frames
                if frame_count % 30 == 0:
                    print(f"\nFrame: {frame_count}")
                    print(f"Road 1: {count1} vehicles | {self.stats_road1.congestion_level} | {signal_status['road1']}")
                    print(f"Road 2: {count2} vehicles | {self.stats_road2.congestion_level} | {signal_status['road2']}")
                
                frame_count += 1
                
                # Check for quit command
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"\nError occurred: {e}")
            raise
        finally:
            # Cleanup
            self.cap_road1.release()
            self.cap_road2.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print("\n" + "=" * 60)
            print("FINAL STATISTICS")
            print("=" * 60)
            print(f"Road 1: Avg vehicles = {self.stats_road1.avg_vehicles_per_frame:.2f}")
            print(f"Road 2: Avg vehicles = {self.stats_road2.avg_vehicles_per_frame:.2f}")
            print("=" * 60)


def main():
    """
    Main function to run the smart traffic system.
    
    Usage:
        Replace 'road1.mp4' and 'road2.mp4' with your actual video files.
    """
    # Video file paths (update these with your actual video paths)
    VIDEO_ROAD1 = "road1.mp4"
    VIDEO_ROAD2 = "road2.mp4"
    
    try:
        # Initialize and run the system
        system = SmartTrafficSystem(VIDEO_ROAD1, VIDEO_ROAD2)
        system.run()
        
    except FileNotFoundError as e:
        print(f"Error: Video file not found - {e}")
        print("\nPlease ensure you have two video files:")
        print("  - road1.mp4: Video of traffic on road 1")
        print("  - road2.mp4: Video of traffic on road 2")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
