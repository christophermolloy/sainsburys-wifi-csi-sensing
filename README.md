# Sainsbury's WiFi CSI Sensing — Basket & Trolley Detection

A proof-of-concept pipeline for detecting and classifying shopping baskets and trolleys at store entrances using WiFi Channel State Information (CSI) sensing.

## Overview

WiFi CSI measures how wireless signals are distorted by objects in the environment. This project uses that signal to classify what's passing through a store entrance: nothing, a person alone, a person with a basket, or a person with a trolley — all without cameras.

```
[WiFi TX] ──── store entrance ────> [WiFi RX]
                    │
              CSI amplitude data
                    │
             CNN classifier
                    │
    empty / person / basket / trolley
```

## Quick Start

### Step 0a — Validate pipeline with synthetic data (no hardware, no download)

```bash
cd csi_sensing
pip install -r requirements.txt

# Trains on synthetic CSI data simulating the 4 entrance classes
python train.py --dataset synthetic --model cnn --epochs 20
```

Expected output: ~85–92% accuracy on the synthetic 4-class problem.  
This confirms the full pipeline (data → model → training → evaluation) works before touching real data.

### Step 0b — Validate on real UT-HAR dataset

```bash
# Download UT-HAR (WiFi human activity recognition dataset, ~130MB)
python data/download.py

# Train on 7-class activity recognition (lie/fall/run/walk/etc.)
python train.py --dataset uthar --model cnn --epochs 50
```

Expected output: ~85–90% test accuracy, matching published SenseFi benchmark results.  
This proves the approach works on real CSI data before you collect your own.

### Step 1 — Evaluate a saved checkpoint

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --dataset synthetic
```

Produces: classification report, confusion matrix plot, per-class CSI heatmaps, inference speed.

---

## Project Structure

```
csi_sensing/
├── train.py                    # Main training entry point
├── evaluate.py                 # Checkpoint evaluation + reporting
├── config.yaml                 # All hyperparameters live here
├── requirements.txt
│
├── data/
│   ├── download.py             # Downloads UT-HAR dataset
│   └── raw/
│       ├── uthar/              # Populated by download.py
│       └── entrance/           # YOUR collected data goes here
│           ├── empty/          # CSV files from empty-zone recordings
│           ├── person_only/
│           ├── person_basket/
│           └── person_trolley/
│
├── src/
│   ├── datasets/
│   │   ├── uthar.py            # UT-HAR loader (Step 0 validation)
│   │   ├── synthetic.py        # Physics-grounded fake CSI (instant testing)
│   │   └── entrance.py         # Loader for your real collected data
│   ├── models/
│   │   ├── cnn.py              # CSICNN + CSIResNet (recommended starting point)
│   │   ├── lstm.py             # Bidirectional LSTM with temporal attention
│   │   └── transformer.py      # Transformer with sinusoidal positional encoding
│   ├── preprocessing/
│   │   └── csi.py              # Amplitude, phase, denoising, windowing, Doppler
│   └── utils/
│       ├── metrics.py          # Accuracy, F1, confusion matrix, early stopping
│       └── visualise.py        # Training curves, CSI heatmaps, class comparisons
│
└── checkpoints/                # Best model weights saved here
```

---

## Configuration

All settings are in `config.yaml`. Key switches:

```yaml
data:
  dataset: synthetic   # ← change to: uthar, entrance
  batch_size: 64

model:
  architecture: cnn    # ← change to: resnet, lstm, transformer
  num_classes: 4       # 4 for synthetic/entrance, 7 for uthar

training:
  epochs: 50
  learning_rate: 0.001
```

Override any config value from the command line:
```bash
python train.py --dataset uthar --model transformer --epochs 100
```

---

## Adding Your Own Collected Data (ESP32 / Raspberry Pi)

When your hardware arrives, collect data as follows:

### 1. Collect CSI with ESP32

```bash
# Flash ESP32-CSI-Tool firmware, then:
# Place ESP32 TX and RX flanking the entrance gate (~2-3m apart)
# Run collection per class:
python -c "
import serial, csv, time
ser = serial.Serial('/dev/tty.usbserial-XXXX', 921600)
with open('data/raw/entrance/person_basket/session_001.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp'] + [f'sc_{i}' for i in range(52)])
    for _ in range(5000):  # ~50 seconds at 100Hz
        line = ser.readline().decode().strip()
        if line.startswith('CSI_DATA'):
            writer.writerow([time.time()] + line.split(',')[1:])
"
```

### 2. Collect CSI with Raspberry Pi (Nexmon)

```bash
# On the Pi (after flashing Nexmon CSI firmware):
sudo nexutil -Isendto:1 -b20 -f5200
sudo tcpdump -i wlan0 dst port 5500 -vv -w capture.pcap

# Transfer capture.pcap to your machine, then parse:
python -c "
import csiread
csidata = csiread.Nexmon('capture.pcap', chip='43455c0', bw=20)
csidata.read()
# csidata.csi → complex (n_frames, n_subcarriers, n_rx_antennas)
import numpy as np
amp = np.abs(csidata.csi)  # extract amplitude
np.save('data/raw/entrance/person_trolley/session_001.npy', amp)
"
```

### 3. Validate data structure

```bash
python -c "
from src.datasets.entrance import EntranceCSIDataset
EntranceCSIDataset.validate_data_dir('./data/raw/entrance')
"
```

### 4. Train on your collected data

```bash
# Edit config.yaml: dataset: entrance, num_classes: 4
python train.py --dataset entrance --model cnn --epochs 50
```

---

## Models

| Model | Best For | Notes |
|-------|----------|-------|
| `cnn` | Quick baseline, production | Fastest inference; ~3ms/sample |
| `resnet` | Highest accuracy | Residual connections help with noisy CSI |
| `lstm` | Sequential patterns | Good for detecting motion trajectories |
| `transformer` | Cross-environment generalisation | Most data-hungry |

Start with `cnn`. Move to `resnet` if accuracy is insufficient.

---

## Expected Results

| Dataset | Model | Expected Accuracy |
|---------|-------|-------------------|
| Synthetic (4-class) | CNN | 85–92% |
| UT-HAR (7-class) | CNN | 83–88% |
| UT-HAR (7-class) | ResNet | 87–92% |
| Entrance (collected) | CNN | 80–90% (depends on data volume) |

Collecting 300–500 labelled passes per class typically yields reliable classification.

---

## Hardware Shopping List (for next steps)

| Item | Cost | Purpose |
|------|------|---------|
| 2× ESP32 DevKit v1 | ~£8 each | Cheapest CSI nodes, 52 subcarriers |
| Raspberry Pi 4 (4GB) | ~£55 | Higher-resolution CSI (256 subcarriers via Nexmon) |
| USB-serial adapter | ~£5 | ESP32 data collection |
| microSD card (32GB) | ~£8 | Pi storage |

---

## References

- [Awesome WiFi CSI Sensing](https://github.com/NTUMARS/Awesome-WiFi-CSI-Sensing) — paper collection
- [SenseFi Benchmark](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) — PyTorch model zoo
- [Nexmon CSI](https://github.com/seemoo-lab/nexmon_csi) — Raspberry Pi firmware
- [ESP32 CSI Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool) — microcontroller collection
- [csiread](https://github.com/citysu/csiread) — fast Python CSI parser
- [UT-HAR Dataset](https://github.com/ermongroup/Wifi_Activity_Recognition) — original dataset
