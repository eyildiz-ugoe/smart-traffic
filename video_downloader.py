"""
Traffic Video Downloader and Setup Script
=========================================
This script helps you download sample traffic videos and set up the environment
for the Smart Traffic Light Automation System.

Author: Traffic Automation Team
Date: October 2025
"""

import urllib.request
import os
from pathlib import Path
import subprocess
import sys


class TrafficVideoSetup:
    """
    Setup helper for downloading traffic videos and installing dependencies.
    """
    
    # Sample traffic video URLs (royalty-free from Pexels/Pixabay)
    VIDEO_URLS = {
        'road1.mp4': [
            'https://www.pexels.com/video/854100/download/',  # Heavy traffic
            'https://www.pexels.com/video/857194/download/',  # Medium traffic
        ],
        'road2.mp4': [
            'https://www.pexels.com/video/3044127/download/',  # Light traffic
            'https://www.pexels.com/video/4812458/download/',  # Variable traffic
        ]
    }
    
    REQUIRED_PACKAGES = [
        'opencv-python',
        'numpy',
    ]
    
    def __init__(self, output_dir: str = '.'):
        """
        Initialize the setup helper.
        
        Args:
            output_dir: Directory to save downloaded videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def install_dependencies(self):
        """
        Install required Python packages.
        """
        print("=" * 60)
        print("Installing Dependencies")
        print("=" * 60)
        
        for package in self.REQUIRED_PACKAGES:
            print(f"\nInstalling {package}...")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', package
                ])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}: {e}")
                return False
        
        print("\n✓ All dependencies installed successfully!")
        return True
    
    def download_video(self, url: str, filename: str) -> bool:
        """
        Download a video from URL.
        
        Args:
            url: Video URL
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        output_path = self.output_dir / filename
        
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, output_path)
            
            # Verify file exists and has content
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"✓ {filename} downloaded successfully")
                return True
            else:
                print(f"✗ Download failed for {filename}")
                return False
                
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            return False
    
    def download_sample_videos(self):
        """
        Download sample traffic videos.
        """
        print("\n" + "=" * 60)
        print("Downloading Sample Traffic Videos")
        print("=" * 60)
        print("\nNote: You can also use your own traffic videos!")
        print("Simply name them 'road1.mp4' and 'road2.mp4'\n")
        
        for filename, urls in self.VIDEO_URLS.items():
            output_path = self.output_dir / filename
            
            # Skip if file already exists
            if output_path.exists():
                print(f"✓ {filename} already exists, skipping download")
                continue
            
            # Try each URL until one works
            success = False
            for url in urls:
                if self.download_video(url, filename):
                    success = True
                    break
            
            if not success:
                print(f"\n⚠ Could not download {filename} automatically")
                print(f"Please download a traffic video manually and save as: {filename}")
    
    def create_test_videos(self):
        """
        Create simple test videos using OpenCV if download fails.
        """
        print("\n" + "=" * 60)
        print("Creating Test Videos")
        print("=" * 60)
        
        try:
            import cv2
            import numpy as np
            
            # Video properties
            width, height = 640, 480
            fps = 30
            duration = 10  # seconds
            
            for i, filename in enumerate(['road1.mp4', 'road2.mp4']):
                output_path = self.output_dir / filename
                
                if output_path.exists():
                    continue
                
                print(f"\nCreating {filename}...")
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                
                # Generate frames
                num_frames = fps * duration
                for frame_num in range(num_frames):
                    # Create blank frame
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    frame[:] = (50, 50, 50)  # Gray background
                    
                    # Draw road
                    cv2.rectangle(frame, (0, height//3), (width, 2*height//3), (80, 80, 80), -1)
                    
                    # Draw lane markings
                    for x in range(0, width, 40):
                        cv2.rectangle(frame, (x, height//2-5), (x+20, height//2+5), (255, 255, 255), -1)
                    
                    # Simulate moving vehicles
                    num_vehicles = 2 + i * 3  # Different density for each road
                    for v in range(num_vehicles):
                        # Calculate vehicle position
                        x = (frame_num * 5 + v * 150) % (width + 100) - 100
                        y = height//2 - 30 + v * 20
                        
                        # Draw vehicle (simple rectangle)
                        cv2.rectangle(frame, (x, y), (x+80, y+40), (0, 0, 255), -1)
                        cv2.rectangle(frame, (x, y), (x+80, y+40), (255, 255, 255), 2)
                    
                    # Add text
                    cv2.putText(frame, f"Road {i+1} - Test Video", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    out.write(frame)
                
                out.release()
                print(f"✓ {filename} created successfully")
            
            print("\n✓ Test videos created successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error creating test videos: {e}")
            return False
    
    def verify_setup(self):
        """
        Verify that all required files exist.
        """
        print("\n" + "=" * 60)
        print("Verifying Setup")
        print("=" * 60)
        
        # Check for video files
        video_files = ['road1.mp4', 'road2.mp4']
        all_exist = True
        
        for filename in video_files:
            path = self.output_dir / filename
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✓ {filename} exists ({size_mb:.2f} MB)")
            else:
                print(f"✗ {filename} not found")
                all_exist = False
        
        return all_exist
    
    def run_full_setup(self):
        """
        Run the complete setup process.
        """
        print("\n" + "=" * 60)
        print("SMART TRAFFIC LIGHT SYSTEM - SETUP")
        print("=" * 60)
        
        # Step 1: Install dependencies
        if not self.install_dependencies():
            print("\n✗ Setup failed: Could not install dependencies")
            return False
        
        # Step 2: Try to download videos (optional)
        print("\n⚠ Note: Automatic video download may not work due to website restrictions")
        print("If downloads fail, the script will create test videos instead\n")
        
        response = input("Attempt to download sample videos? (y/n): ").lower()
        if response == 'y':
            self.download_sample_videos()
        
        # Step 3: Create test videos if needed
        if not self.verify_setup():
            print("\n⚠ Video files not found. Creating test videos...")
            self.create_test_videos()
        
        # Step 4: Final verification
        if self.verify_setup():
            print("\n" + "=" * 60)
            print("✓ SETUP COMPLETE!")
            print("=" * 60)
            print("\nYou can now run the traffic light system:")
            print("  python smart_traffic_system.py")
            print("\n" + "=" * 60)
            return True
        else:
            print("\n✗ Setup incomplete. Please ensure video files exist.")
            return False




def print_manual_setup_guide():
    """Print detailed instructions for manual setup."""
    print("\n" + "=" * 60)
    print("MANUAL SETUP GUIDE")
    print("=" * 60)
    print(
        "If automatic setup fails, follow these steps:\n\n"
        "1. Install Dependencies:\n"
        "   pip install opencv-python numpy\n\n"
        "2. Get Traffic Videos:\n"
        "   - Download two contrasting clips that show different traffic densities.\n"
        "   - Recommended samples (royalty free):\n"
        "       * https://www.pexels.com/video/854100/\n"
        "       * https://www.pexels.com/video/3044127/\n"
        "   - Save them next to this script as road1.mp4 and road2.mp4.\n\n"
        "3. Run the Simulation:\n"
        "   python smart_traffic_system.py\n\n"
        "4. (Optional) Record Metrics:\n"
        "   - Observe the console logs to verify green light time adjustments.\n"
        "   - Capture screenshots for documentation if needed.\n\n"
        "Happy experimenting!"
    )
    print("=" * 60)


def main():
    """Entry point for the setup helper when executed as a script."""
    setup = TrafficVideoSetup()

    if not setup.run_full_setup():
        print("\nAutomatic setup was not successful. Showing manual guide...")
        print_manual_setup_guide()


if __name__ == "__main__":
    main()
