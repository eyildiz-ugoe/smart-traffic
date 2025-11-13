"""Command line entry point for the adaptive traffic light system."""

from __future__ import annotations

import argparse
import logging

from adaptive_traffic import TrafficConfig, TrafficSystem

logging.basicConfig(level=logging.INFO)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["realistic", "simulation"], default="simulation")
    parser.add_argument("--video-a", dest="video_path_road_a", help="Video path for road A")
    parser.add_argument("--video-b", dest="video_path_road_b", help="Video path for road B")
    parser.add_argument("--spawn-rate-a", type=float, default=12.0)
    parser.add_argument("--spawn-rate-b", type=float, default=12.0)
    parser.add_argument("--min-green", type=float, default=5.0)
    parser.add_argument("--max-red", type=float, default=60.0)
    parser.add_argument("--yellow", type=float, default=3.0)
    parser.add_argument("--zone", type=float, default=0.25, help="Detection zone height fraction")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = TrafficConfig(
        mode=args.mode,
        video_path_road_a=args.video_path_road_a,
        video_path_road_b=args.video_path_road_b,
        spawn_rate_road_a=args.spawn_rate_a,
        spawn_rate_road_b=args.spawn_rate_b,
        min_green_time=args.min_green,
        max_red_time=args.max_red,
        yellow_time=args.yellow,
        detection_zone_size=args.zone,
        detection_confidence=args.confidence,
    )
    system = TrafficSystem(config)
    system.run()


if __name__ == "__main__":
    main()
