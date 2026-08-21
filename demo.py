"""Pilot demo launcher — three adaptive-signal cases, simulation and real.

Case 1  Pedestrian crossing   one road + crosswalk; cars keep green until a
                              pedestrian actually needs to cross.
Case 2  Two-road intersection two camera feeds / two synthetic roads with
                              queue-pressure-based green extension.
Case 3  Four-way intersection crossroads with two-phase adaptive control;
                              empty approaches never hold up the others.

Examples
--------
python demo.py --list
python demo.py --case 1 --mode simulation --fullscreen
python demo.py --case 1 --mode real
python demo.py --case 2 --mode simulation --seed 42
python demo.py --case 2 --mode real
python demo.py --case 3 --mode simulation
python demo.py --case 3 --mode real --video videos/intersection.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("demo")

CASES = {
    1: (
        "Pedestrian crossing",
        "Cars stay green until a pedestrian is detected; walk phase is "
        "granted at the next safe gap (dilemma-zone guard).",
    ),
    2: (
        "Two-road intersection",
        "Adaptive green split between two one-way feeds based on live "
        "vehicle counts and queue pressure.",
    ),
    3: (
        "Four-way intersection",
        "Two-phase crossroads control; an empty axis is skipped so the "
        "cross traffic proceeds without waiting out a fixed plan.",
    ),
}

# Urban Tracker research sequences (Jodoin et al., WACV 2014): fixed
# elevated cameras with clearly detectable vehicles AND pedestrians.
# The aerial Mixkit clips (videos/pedestrian.mp4, videos/intersection.mp4)
# are kept as presentation b-roll; they are filmed too high for reliable
# YOLOv8n detection.
DEFAULT_REAL_VIDEOS = {
    1: "videos/rouen_crosswalk.avi",
    3: "videos/sherbrooke_intersection.avi",
}


def _print_cases() -> None:
    print("Available pilot demo cases:\n")
    for number, (title, description) in CASES.items():
        print(f"  Case {number}: {title}")
        print(f"          {description}")
        modes = "simulation | real"
        print(f"          modes: {modes}\n")


def _ensure_case_video(case: int, override: str | None) -> Path:
    """Resolve the video for a real-mode case, downloading it if needed."""

    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        return path

    path = Path(DEFAULT_REAL_VIDEOS[case])
    if path.exists():
        return path

    from video_downloader import ensure_video

    downloaded = ensure_video(path.name, output_dir=str(path.parent))
    if downloaded is None:
        raise FileNotFoundError(
            f"{path} is missing and could not be downloaded automatically. "
            "See README 'Demo videos' for manual sources."
        )
    return Path(downloaded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smart traffic pilot demo (three cases, simulation + real)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case", type=int, choices=sorted(CASES), help="Demo case to run")
    parser.add_argument(
        "--mode",
        choices=["simulation", "real"],
        default="simulation",
        help="Synthetic traffic or prerecorded video with YOLO detection",
    )
    parser.add_argument("--video", type=str, help="Video override for real mode (cases 1 and 3)")
    parser.add_argument("--video-road1", type=str, help="Case 2 real mode: video for road 1")
    parser.add_argument("--video-road2", type=str, help="Case 2 real mode: video for road 2")
    parser.add_argument("--fps", type=int, default=30, help="Simulation frame rate")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible simulations")
    parser.add_argument("--max-frames", type=int, help="Stop after N frames (default: run until 'q')")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no GUI window)")
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen (press F to toggle)")
    parser.add_argument("--list", action="store_true", help="List the demo cases and exit")
    return parser


def run_case1(args: argparse.Namespace) -> None:
    if args.mode == "simulation":
        from pedestrian_crossing import PedestrianCrossingSimulation

        sim = PedestrianCrossingSimulation(fps=args.fps, seed=args.seed)
        sim.run(
            max_frames=args.max_frames,
            display_window=not args.no_display,
            fullscreen=args.fullscreen and not args.no_display,
        )
    else:
        from pedestrian_crossing import RealPedestrianCrossing

        video = _ensure_case_video(1, args.video)
        demo = RealPedestrianCrossing(video)
        demo.run(
            max_frames=args.max_frames,
            display_window=not args.no_display,
            fullscreen=args.fullscreen and not args.no_display,
        )


def run_case2(args: argparse.Namespace) -> None:
    if args.mode == "simulation":
        from smart_traffic_system import SimulationTrafficSystem

        sim = SimulationTrafficSystem(fps=args.fps, seed=args.seed)
        sim.run(
            max_frames=args.max_frames,
            display_window=not args.no_display,
            fullscreen=args.fullscreen and not args.no_display,
        )
    else:
        from smart_traffic_system import SmartTrafficSystem, resolve_video_sources

        video1, video2 = args.video_road1, args.video_road2
        if bool(video1) != bool(video2):
            raise SystemExit(
                "Case 2 real mode needs both --video-road1 and --video-road2 "
                "(or neither, to use the bundled/default videos)."
            )
        if not video1 or not video2:
            video1, video2 = resolve_video_sources()
        system = SmartTrafficSystem(video1, video2)
        system.run(
            max_frames=args.max_frames,
            display_window=not args.no_display,
            fullscreen=args.fullscreen and not args.no_display,
        )


def run_case3(args: argparse.Namespace) -> None:
    if args.mode == "simulation":
        from four_way_intersection import FourWaySimulation

        sim = FourWaySimulation(fps=args.fps, seed=args.seed)
        sim.run(
            max_frames=args.max_frames,
            display_window=not args.no_display,
            fullscreen=args.fullscreen and not args.no_display,
        )
    else:
        from four_way_intersection import RealFourWayIntersection

        video = _ensure_case_video(3, args.video)
        demo = RealFourWayIntersection(video)
        demo.run(
            max_frames=args.max_frames,
            display_window=not args.no_display,
            fullscreen=args.fullscreen and not args.no_display,
        )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list or args.case is None:
        _print_cases()
        if args.case is None and not args.list:
            parser.error("--case is required (or use --list to see the options)")
        return

    title, _ = CASES[args.case]
    logger.info("Running Case %d (%s) in %s mode", args.case, title, args.mode)
    {1: run_case1, 2: run_case2, 3: run_case3}[args.case](args)


if __name__ == "__main__":
    main()
