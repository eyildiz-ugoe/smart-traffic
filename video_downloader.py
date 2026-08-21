"""
Traffic Video Downloader and Setup Script
=========================================
This script helps you download sample traffic videos and set up the environment
for the Smart Traffic Light Automation System.

Author: Traffic Automation Team
Date: October 2025
"""

import hashlib
import urllib.request
import os
from pathlib import Path
import subprocess
import sys


class TrafficVideoSetup:
    """
    Setup helper for downloading traffic videos and installing dependencies.
    """
    
    # Royalty-free sample videos with direct-download URLs (verified 2026-08).
    #
    # road1/road2 ship with the repository via Git LFS. Pexels removed its
    # unauthenticated /download/ endpoints, so those two now fall back to the
    # synthesized test videos when absent; see README "Demo videos" for
    # manually downloadable sources.
    #
    # pedestrian.mp4  -> Mixkit "Aerial view of people at pedestrian crossing"
    #                    (Mixkit Free License), fallback: Urban Tracker "Rouen"
    #                    research sequence (Jodoin et al., WACV 2014).
    # intersection.mp4 -> Mixkit "Busy intersection aerial view" (Mixkit Free
    #                    License), fallback: Urban Tracker "Sherbrooke".
    VIDEO_URLS = {
        'road1.mp4': [],
        'road2.mp4': [],
        # Case 1 real mode: elevated crosswalk view, pedestrians + vehicles.
        'rouen_crosswalk.avi': [
            'https://www.jpjodoin.com/urbantracker/dataset/rouen/rouen_video.avi',
        ],
        # Case 3 real mode: fixed camera over a four-way intersection.
        'sherbrooke_intersection.avi': [
            'https://www.jpjodoin.com/urbantracker/dataset/sherbrooke/sherbrooke_video.avi',
        ],
        # Presentation b-roll (aerial views; too high for reliable YOLOv8n).
        'pedestrian.mp4': [
            'https://assets.mixkit.co/videos/61/61-720.mp4',
        ],
        'intersection.mp4': [
            'https://assets.mixkit.co/videos/60/60-720.mp4',
        ],
    }
    
    #: Expected SHA-256 of pinned downloads (computed 2026-08-21). A hash
    #: mismatch fails the download closed rather than feeding unexpected
    #: bytes from a third-party host into the video decoders.
    VIDEO_SHA256 = {
        'rouen_crosswalk.avi':
            '1604fb168ef88e8fc1cde7be28416433bfff9cf23c3053eebb107edb39238bdd',
        'sherbrooke_intersection.avi':
            'ead4eada5a281e6a5054ea325abe939549e6440807a27c731aa9d54a4a0b71ef',
        'pedestrian.mp4':
            'baed59da05bc93811440556f42d7f8b475c169563ce9504f84382caa91b7a76a',
        'intersection.mp4':
            'c3b679464744970b9521a9c3c14e2f793e142a492d6e81c6667bae2a64f5b6de',
    }

    #: Hard ceiling for any single download; the pinned videos are <= 25 MB.
    MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024

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
        # Never let a caller-supplied name escape the output directory.
        filename = Path(filename).name
        output_path = self.output_dir / filename

        # Stream into a temporary file and move it into place atomically, so
        # an interrupted transfer can never leave a corrupt half-video that
        # later runs would mistake for a valid download. Size is capped and,
        # where a pin is known, the SHA-256 must match before acceptance.
        partial_path = output_path.with_name(output_path.name + '.part')
        expected_hash = self.VIDEO_SHA256.get(filename)
        try:
            print(f"Downloading {filename}...")
            request = urllib.request.Request(
                url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            digest = hashlib.sha256()
            received = 0
            with urllib.request.urlopen(request, timeout=120) as response, \
                    open(partial_path, 'wb') as out_file:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > self.MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"download exceeds {self.MAX_DOWNLOAD_BYTES} bytes"
                        )
                    digest.update(chunk)
                    out_file.write(chunk)

            if received == 0:
                print(f"✗ Download failed for {filename} (empty response)")
                return False
            if expected_hash is not None and digest.hexdigest() != expected_hash:
                print(
                    f"✗ {filename}: checksum mismatch (got {digest.hexdigest()[:16]}…) — "
                    "refusing the file; the upstream source may have changed."
                )
                return False

            os.replace(partial_path, output_path)
            print(f"✓ {filename} downloaded successfully")
            return True

        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            return False
        finally:
            if partial_path.exists():
                try:
                    partial_path.unlink()
                except OSError:
                    pass
    
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




def is_plausible_video(path) -> bool:
    """Cheap sanity check that ``path`` holds real video data.

    Rejects missing/empty files, tiny stubs, Git-LFS pointer files (which
    appear when a repository is cloned without ``git lfs``; ~130 bytes of
    text starting with ``version https://git-lfs``), and HTML/JSON error
    pages served with a 200 status.
    """

    path = Path(path)
    try:
        if not path.exists() or path.stat().st_size < 100 * 1024:
            return False
        with open(path, 'rb') as handle:
            head = handle.read(64)
    except OSError:
        return False
    if head.startswith(b'version https://git-lfs'):
        return False
    if head.lstrip()[:1] in (b'<', b'{'):
        return False
    return True


def ensure_video(filename: str, output_dir: str = 'videos'):
    """Return the path to ``filename``, downloading it if a source is known.

    Existing files are validated first (guarding against truncated downloads
    and Git-LFS pointer stubs); invalid files are re-downloaded. Returns the
    path as a string on success, otherwise ``None``.
    """

    filename = Path(filename).name
    setup = TrafficVideoSetup(output_dir)
    target = setup.output_dir / filename
    if is_plausible_video(target):
        return str(target)

    for url in TrafficVideoSetup.VIDEO_URLS.get(filename, []):
        if setup.download_video(url, filename) and is_plausible_video(target):
            return str(target)
    return None


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
