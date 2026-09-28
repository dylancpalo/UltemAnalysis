# Mu2e tracker Ultem analysis and machining quality control

This repository contains image-analysis and comparison tools for the Ultem mounting components used in tracker-station assembly. It supports quality control of the machined and assembled parts and helps determine whether the Ultem geometry and machining approach meet the positioning needs of the tracker.

## Workflow role

1. Capture calibration-disk and Ultem images before and after epoxy installation.
2. Analyze the images to measure tooling-ball positions in the camera coordinate system.
3. Compare the Ultem measurements with the calibration-disk reference and the expected machined values.
4. Review the offsets, spreads, and before/after changes to assess part quality and inform the Ultem machining approach.

The Ultem work is a two-part task: it checks the quality of the parts and supports decisions about how the Ultems should be machined. These measurements complement the station, panel, X-ray, and tracker-frame metrology combined in [MetrologyMain](https://github.com/dylancpalo/MetrologyMain).

## Contents

- `UltemImageAnalysis.py` and `ImageAnalysis.py` — image processing and tooling-ball measurement routines.
- `StationCircleFit.py` — station/circle fitting utilities.
- `StationUltemAlignmentGlobal.py` — global Ultem alignment analysis.
- `UltemResultsComp.py` — result comparison tools.
- `SortingCSV.py` — CSV sorting utility.
- [Ultem installation summary (Mu2e DocDB document 57619)](https://mu2e-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=57619) — installation, imaging, epoxy, analysis, and station-lowering procedures.

The analysis expects image data and measurement spreadsheets/CSV files prepared for the corresponding scripts. Check each script's arguments and analysis parameters against the measurement setup before processing a new dataset.

## Related repositories

- [CameraMetrologyImageGrab_Processing](https://github.com/dylancpalo/CameraMetrologyImageGrab_Processing) covers camera-image acquisition and processing during station assembly.
- [MetrologyMain](https://github.com/dylancpalo/MetrologyMain) combines camera, X-ray, tracker-frame, and mechanical survey inputs and performs offline fits for alignment tables.
- [TrackBasedAlignment](https://github.com/dylancpalo/TrackBasedAlignment) uses reconstructed tracks to study and validate the alignment.
