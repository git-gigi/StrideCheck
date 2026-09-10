# StrideCheck

Running form analysis system.

## Overview
StrideCheck takes a video of a runner (e.g. on a treadmill) and outputs an analysis report. It evaluates form and gives a score based on biomechanics using MediaPipe. 

It was built because I had knee issues from bad form and wanted an accessible tool instead of paying for clinical equipment.

## Tech Stack
- **Pose**: MediaPipe
- **Processing**: SciPy (Savitzky-Golay) + NumPy + Pandas
- **UI**: Streamlit + Plotly

## Quickstart

```bash
# Clone and setup env
git clone https://github.com/YOUR_USERNAME/StrideCheck.git
cd StrideCheck
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run
streamlit run app.py
```

## Architecture
The system is built as a linear pipeline, but uses a Strategy Pattern for the analyzer so we can swap it later without breaking the rest of the code.

`Video MP4 -> PoseExtractor -> AngleCalculator -> SignalFilter -> GaitSegmenter -> FeatureExtractor -> FormAnalyzer -> Streamlit UI`

Currently (v1) `FormAnalyzer` uses heuristic thresholds from literature. Next step is an XGBoost model. 

## Errors Detected
- Overstriding (Risk: Shin splints, knee stress)
- Low Cadence (< 160 spm)
- Excessive Bounce (energy waste)
- Forward lean (> 10°)
- Asymmetry
- Insufficient Knee Drive

## Roadmap & Future Work
- **XGBoost Classifier**: Swap the current rule-based heuristic analyzer with an ML model trained on real-world running data, leveraging the existing Strategy Pattern.
- **Multi-exercise Support**: Add analysis for other movements (squats, lunges) using a CNN router.
- **Mobile App**: Convert the processing pipeline to TFLite for on-device inference.

## License
MIT