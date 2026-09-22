# Demo guide and live UI screenshots

[Open the five-page demo PDF](DEMO_GUIDE.pdf).

These screenshots were captured from the working studio on the remote computer on 2026-09-22. They show the verified **Heavy rain at night** experiment: source Clear daylight, strength 0.7, seed 42 (effective strength 0.875).

## Generated video
![Generated video and controls](screenshots/studio-generated.png)

## Before and after
![Recorded daylight and synthetic rain at night](screenshots/studio-comparison.png)

## All nine sensor streams
![Generated native sensor previews](screenshots/studio-sensors.png)

The verified output comprises 48 frames (4.806 seconds), 432 primary sensor files, 88,286,126 events and 3,828,874 points. The full native sensor archive is approximately 586 MB and remains in the remote job directory.

**Model role:** sensor-specific project code creates weather and lighting augmentations. V-JEPA 2.1 encodes source and generated visualizations and measures representation changes; it is not a text-to-video or raw-sensor decoder. The augmentation is a research prototype, not a physically calibrated sensor simulator.

The PDF covers operation, native formats, output locations, installation, validation, limitations and upstream sources. Model weights, full datasets and job archives are downloaded or generated separately.
