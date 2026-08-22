# Demo Script — Traffic Department Presentation

Rehearsal-ready run order. Every simulation uses a fixed seed, so what you
rehearse is exactly what plays on stage. On-screen controls in every
window: **SPACE** pause (freeze the frame while you talk), **F**
fullscreen, **Q** end (shows the KPI card), **ESC** hard exit.

Before the meeting: run each command once on the presentation machine.
Total runtime ≈ 12–15 minutes including talking.

---

## Act 1 — The idea, at its simplest (≈3 min)

```bash
python demo.py --case 1 --mode simulation --fullscreen --seed 42
```

- Let it run: **no pedestrian → the car light simply never turns red.**
- When a pedestrian appears, press **SPACE**: point at the dilemma-zone
  message — "it will not start the yellow while a fast car cannot stop."
- Resume; watch yellow → all-red → WALK; cars held behind the line.
- Press **Q**: the KPI card shows walk phases served and average wait.

Say: *"Every feature you will see is a version of this one idea: the
signal only takes time from someone when it gives it to someone else."*

## Act 2 — Measured demand, not a plan (≈2 min)

```bash
python demo.py --case 2 --mode simulation --fullscreen --seed 42
```

- Point at the queue-pressure numbers and the per-vehicle priority boxes.
- The green extends under load and rests in green when the other road is
  empty. Press **Q** for the switches/vehicles KPI card.

## Act 3 — The crossroads (≈3 min)

```bash
python demo.py --case 3 --mode simulation --fullscreen --seed 7
```

- Full approach names and direction arrows — no jargon on screen.
- Watch "Switches: N (demand-driven: M)" — in rehearsals every switch is
  demand-driven; say so when it shows.
- **SPACE** during a yellow: walk through green → yellow → all-red.
- **Q** for the card: *"an empty road never holds a green hostage."*

## Act 4 — Same brain, real cameras (≈4 min)

```bash
python demo.py --case 3 --mode real
```

- Sherbrooke, Montreal — a real intersection, live YOLOv8 on every frame.
- Point at the zone labels, the PARKED tags on the curbside cars, the
  parked counter in the panel: *"a parked car never counts as demand."*
- Explain shadow mode: detections are real; the plan is displayed, not
  actuated — exactly how a pilot starts on your cameras.

## Act 5 — And in the rain (≈2 min, requires the RainSnow dataset)

```bash
python demo.py --case 3 --mode real --preset rain
```

- Rainy Danish intersection; zones were **learned by the system itself**
  (auto-calibration) — mention no one drew them.
- Fallback if the dataset is missing on the machine: show
  `docs/auto_calibration_rainsnow.jpg` and `docs/rainsnow_night_rain_challenge.jpg`
  instead, and quote the measured baseline from the README.

## Closing numbers to say out loud

- Pittsburgh's Surtrac (same class of system): **−26 % travel time,
  −41 % idling, −21 % emissions** in its pilot.
- Vienna runs camera-based pedestrian signals in production since 2018.
- This system: three cases, simulation + real footage, 86 automated
  tests, safety fuzz-tested over millions of inputs, parked-car immunity
  and self-calibration demonstrated on peer-reviewed datasets.
- The ask: **Phase 1 — shadow deployment on existing city cameras.
  Software only. Measured numbers in three months.**

## Unattended booth / lobby screen

```bash
python demo.py --kiosk               # cycles all acts fullscreen, forever
python demo.py --kiosk --kiosk-seconds 45
```

ESC exits; Q skips a segment. Fresh random seeds each cycle so it never
looks canned.
