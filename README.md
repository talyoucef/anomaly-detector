# Telecom Radio Signal Anomaly Detection

This project simulates a realistic cellular network environment and detects radio signal anomalies using unsupervised machine learning. It bridges telecommunications engineering and data science by modeling network conditions, specifically **RSSI** (Received Signal Strength Indicator) and **SNR** (Signal-to-Noise Ratio), to automatically identify hardware failures and external interference.

## The Telecom Use Case

In modern cellular networks, monitoring radio conditions is critical for maintaining Quality of Service (QoS). This project generates a synthetic time-series dataset of RSSI and SNR values and injects domain-specific anomalies:

* **Antenna degradation / failures:** sudden, severe drops in RSSI.
* **External interference:** spikes in noise that degrade SNR without a proportional drop in RSSI.

An **Isolation Forest** algorithm is then applied to automatically isolate these degraded states from normal network operation.

## Project Structure

```text
anomaly-detector/
├── experiments/
│   ├── exploration.ipynb
│   └── isolation\\\_forest\\\_test.ipynb
├── generate\\\_signals.py
├── isolation\\\_forest\\\_results.png
├── model\\\_isolation\\\_forest.py
├── README.md
└── signal\\\_data.csv
```

## Getting Started

### Prerequisites

Make sure Python is installed, then install the required libraries:

```bash
pip install pandas scikit-learn matplotlib
```

### Running the Project

**1. Generate the synthetic network data**

```bash
python generate\\\_signals.py
```

This creates a CSV file containing the simulated RSSI and SNR metrics over time.

**2. Run the anomaly detection model**

```bash
python model\\\_isolation\\\_forest.py
```

This processes the dataset, trains the Isolation Forest model, evaluates its accuracy, and generates a visualization of the detected anomalies.

## Results

The Isolation Forest successfully isolated the degraded signal conditions, clearly distinguishing normal network behavior from both interference and equipment failure.

!\[Isolation Forest anomaly detection results](isolation\_forest\_results.png)

## Experiments Folder

The `experiments/` directory serves as a scratchpad for initial research and contains two Jupyter notebooks: `exploration.ipynb` and `isolation\\\_forest\\\_test.ipynb`. They were used to compare approaches to setting the contamination rate, such as calculating it dynamically with a 3-sigma threshold (based on the signal's mean and standard deviation) versus hardcoding it, before the final logic was integrated into the main Python scripts.

## Technical Notes \& Future Improvements

* **Contamination threshold:** the `contamination` parameter of the Isolation Forest is set to `0.05`, matching the known anomaly injection rate. This is a deliberate shortcut for evaluation purposes. In a blind, real-world scenario where the true anomaly rate is unknown, this threshold would need manual tuning and validation by a radio optimization engineer.
* **Rolling averages (planned):** future iterations will add time-series rolling averages for both RSSI and SNR. Analyzing the rate of change over time, rather than only point-in-time values, should help the model detect slow, progressive hardware degradation before a complete failure occurs.

