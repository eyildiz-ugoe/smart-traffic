# Smart Traffic Light Automation

This project simulates a smart two-way intersection that adapts the green light
allocation based on the detected number of vehicles on each road. Vehicle
density is now obtained from a YOLOv8 detector running on real traffic footage
and the counts feed directly into an adaptive state machine.

## Features

- Real-time vehicle counting backed by the Ultralytics YOLOv8 models.
- Dynamic green-light duration based on congestion levels.
- Minimal visual overlay – only the virtual traffic lights are rendered on top
  of the original footage.
- Helper script to download or synthesise traffic clips for experimentation.
- Curated scenarios that mirror contrasting traffic patterns for reproducible
  tests.

## Getting Started

1. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```

   The Ultralytics package will install PyTorch automatically if it is not yet
   present. See the [official docs](https://docs.ultralytics.com) for GPU setup
   tips.

2. **Download YOLO weights**

   The system defaults to the small `yolov8n.pt` model. Place the file in the
   project root (or provide a custom path when instantiating
   `SmartTrafficSystem`). Pretrained weights are available from the
   [Ultralytics model zoo](https://github.com/ultralytics/ultralytics#models).

3. **Prepare videos**

   - Recommended royalty-free samples used during development:
     - [Pexels 854100 – heavy arterial traffic](https://www.pexels.com/video/854100/)
     - [Pexels 3044127 – lighter opposing traffic](https://www.pexels.com/video/3044127/)
   - Save them in the project root as `road1.mp4` and `road2.mp4`.
   - Alternatively run the setup helper:

     ```bash
     python video_downloader.py
     ```

     The helper attempts to download the same clips and will fall back to
     generating synthetic videos if direct downloads are blocked.

4. **Run the simulation**

   ```bash
   python smart_traffic_system.py
   ```

   Press `q` to exit the live window.

## Scenarios

Predefined scenarios are available in `traffic_scenarios.py`. They encode the
vehicle counts observed in the recommended videos, as well as an alternating
surge pattern. You can import and iterate over them for offline tests or to
script automated evaluations.

```python
from traffic_scenarios import load_predefined_scenarios

for scenario in load_predefined_scenarios():
    print(scenario.name)
    for road1, road2 in scenario.steps():
        # Feed counts into TrafficLightController for dry runs
        ...
```

## Testing

Automated tests cover the adaptive controller, statistics tracking, and scenario
validation.

```bash
python -m pytest
```

Always run the tests before committing new changes.
