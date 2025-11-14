# Smart Traffic Light Automation 🚦

An intelligent two-way intersection system that adapts traffic light timing based on real-time vehicle detection using YOLOv8. The system supports both **real video mode** (with actual traffic footage) and **simulation mode** (synthetic traffic generation without video files).

## ✨ Features

### Core Capabilities
- **Real-time vehicle detection** using Ultralytics YOLOv8
- **Queue-aware ranking** - sorts vehicles by proximity to stop line
- **Dynamic traffic light control** based on congestion and queue pressure
- **Two operation modes**:
  - 🎥 **Real Mode**: Process actual traffic video footage
  - 🚗 **Simulation Mode**: Generate synthetic traffic (no videos needed!)
- **Rich visualization** with bounding boxes, traffic lights, and live statistics
- **Automated testing** with comprehensive test suite

### Visualization Features
1. Color-coded traffic signals (RED, YELLOW, GREEN)
2. Vehicle bounding boxes with priority numbers
3. Queue threshold lines (approach and exit zones)
4. Real-time statistics overlay:
   - Vehicle count
   - Queue pressure
   - Congestion level
   - Stopline status
5. Fullscreen-ready viewer (press `F` to toggle) for immersive demos

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the System

**Simulation Mode (NO Videos Required!) ⭐ Easiest option**

```bash
# Basic simulation
python smart_traffic_system.py --mode simulation

# Customize simulation
python smart_traffic_system.py --mode simulation --spawn-rate 5.0 --max-frames 100

# Headless mode (no display)
python smart_traffic_system.py --mode simulation --no-display --max-frames 100

# Reproducible results
python smart_traffic_system.py --mode simulation --seed 42
```

**Real Video Mode**

```bash
# Auto-downloads videos if missing
python smart_traffic_system.py

# Or specify videos
python smart_traffic_system.py --video-road1 videos/road1.mp4 --video-road2 videos/road2.mp4

# Download videos manually (optional)
python video_downloader.py
```

**Recommended video sources:**
- [Pexels 854100 – heavy traffic](https://www.pexels.com/video/854100/)
- [Pexels 3044127 – lighter traffic](https://www.pexels.com/video/3044127/)

Press `q` to exit any visualization window.

## 📋 All Command-Line Options

```bash
python smart_traffic_system.py [OPTIONS]

Options:
  --mode {real,simulation}          Operation mode (default: real)
  --video-road1 PATH                Video file for road 1 (real mode)
  --video-road2 PATH                Video file for road 2 (real mode)
  --orientation-road1 {vertical,horizontal}  Camera orientation road 1
  --orientation-road2 {vertical,horizontal}  Camera orientation road 2
  --fps INT                         Simulation frame rate (default: 30)
  --spawn-rate FLOAT                Vehicles per second (default: 3.5)
  --spawn-rate-road1 FLOAT          Override spawn rate for road 1 (simulation)
  --spawn-rate-road2 FLOAT          Override spawn rate for road 2 (simulation)
  --max-frames INT                  Limit frames in simulation
  --seed INT                        Random seed for reproducibility
  --no-display                      Run without GUI window
  --fullscreen                      Launch simulation UI fullscreen (press F to toggle)
  -h, --help                        Show help message
```

### Example Commands

```bash
# Simulation with custom settings
python smart_traffic_system.py --mode simulation --fps 30 --spawn-rate 4.0 --seed 42

# Launch fullscreen immediately
python smart_traffic_system.py --mode simulation --fullscreen

# Real mode with specific videos
python smart_traffic_system.py --video-road1 path/to/video1.mp4 --video-road2 path/to/video2.mp4

# Quick test (100 frames, no display)
python smart_traffic_system.py --mode simulation --max-frames 100 --no-display
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/test_animation_run.py -v
```

### Individual Tests
```bash
# Simulation mode (no videos needed)
pytest tests/test_animation_run.py::test_simulation_mode -v -s

# Real video processing
pytest tests/test_animation_run.py::test_system_runs_with_real_videos -v -s

# Video looping
pytest tests/test_animation_run.py::test_system_video_loop -v -s

# Visualization components
pytest tests/test_animation_run.py::test_visualization_components -v -s
```

All tests (4/4) should pass ✅

## 📦 Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
pytest>=7.4.0
ultralytics>=8.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

**Note**: Ultralytics will auto-install PyTorch. See [official docs](https://docs.ultralytics.com) for GPU setup.



## 📊 System Architecture

### Core Components

**`SmartTrafficSystem`** - Real video processing
- Video capture and frame processing
- YOLOv8 vehicle detection
- Queue analysis and metrics
- Traffic light control
- Visualization rendering

**`SimulationTrafficSystem`** - Synthetic traffic generation
- Procedural vehicle spawning
- Realistic movement simulation
- No video files required
- Ideal for testing and demos

**`VehicleDetector`** - YOLOv8 wrapper
- Configurable confidence thresholds
- Multiple vehicle class support
- GPU/CPU automatic fallback

**`VehicleQueueAnalyzer`** - Queue metrics
- Distance-based vehicle sorting
- Queue pressure calculation
- Stopline occupancy detection
- Exit zone monitoring

**`TrafficLightController`** - Adaptive timing
- Dynamic green phase extension
- Queue pressure-based decisions
- Minimum/maximum timing constraints
- Smooth yellow transitions

### Key Modules

- `traffic_core.py` - Controller and statistics
- `counter.py` - Vehicle counting logic
- `sorter.py` - Queue prioritization
- `video_downloader.py` - Video setup helper
- `traffic_scenarios.py` - Predefined test scenarios

## 🎯 Use Cases

### Development & Testing
```python
from smart_traffic_system import SimulationTrafficSystem

# Quick test with synthetic traffic
sim = SimulationTrafficSystem(fps=30, seed=42)
sim.run(max_frames=100, display_window=False)

# Access statistics
print(f"Avg vehicles: {sim.stats_road1.avg_vehicles_per_frame:.2f}")
```

### Production with Real Videos
```python
from smart_traffic_system import SmartTrafficSystem

system = SmartTrafficSystem(
    video_road1="videos/road1.mp4",
    video_road2="videos/road2.mp4",
    orientation_road1="vertical",
    orientation_road2="vertical"
)
system.run()
```

### Offline Scenario Testing
```python
from traffic_scenarios import load_predefined_scenarios

for scenario in load_predefined_scenarios():
    print(f"Testing scenario: {scenario.name}")
    for road1_count, road2_count in scenario.steps():
        # Feed into controller
        pass
```

## 🔧 Configuration

### Detector Settings
```python
from smart_traffic_system import DetectorConfig

config = DetectorConfig(
    model_path="weights/yolov8n.pt",
    confidence=0.25,
    iou=0.5,
    classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
    device="cuda",  # or "cpu"
    max_detections=100
)
```

### Queue Analyzer Settings
```python
analyzer = VehicleQueueAnalyzer(
    orientation="vertical",
    approach_threshold_ratio=0.65,
    exit_margin=5
)
```

## 🐛 Troubleshooting

### Video files not found
The system auto-generates test videos or downloads samples automatically.

### YOLO weights missing
YOLOv8n weights (~6MB) download automatically on first run to `weights/yolov8n.pt`.

### Display window doesn't appear
- Check OpenCV GUI support
- Try `--no-display` flag for headless mode
- On remote servers, ensure X11 forwarding or use headless mode

### Slow performance
- **With GPU**: ~30-60 FPS
- **CPU only**: ~5-15 FPS
- Use smaller model (yolov8n.pt) instead of yolov8x.pt
- Reduce video resolution
- Lower spawn rate in simulation mode

## 📈 Performance Notes

- First run downloads YOLOv8 weights (~6MB for yolov8n)
- GPU acceleration automatically detected
- Videos loop automatically at end
- Statistics computed in real-time
- Headless mode for automated testing

## 🤝 Contributing

1. Run tests before committing: `pytest`
2. Follow existing code style
3. Add tests for new features
4. Update documentation

## 📄 License

See LICENSE file for details.

---

**Pro tip:** Start with simulation mode (`--mode simulation`) - no setup required! 🚀
