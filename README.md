# CCTV Human Detection System

This project implements a real-time human detection system using YOLOv8 and DeepSORT. The system is designed to monitor CCTV feeds and send alerts if human activity is detected after a specified cutoff time.

## Features
- Real-time human detection using YOLOv8 (GPU-accelerated)
- Object tracking with DeepSORT to prevent false positives
- Automatic email alerts for unauthorized human detection
- Configurable cutoff time and cooldown periods

## Installation

### 1. Clone the Repository
git clone https://github.com/Arnav-chib/cctv_human.git cd cctv-monitor


### 2. Set up the Environment
Create a Python virtual environment and install dependencies:

```bash
conda create -n cctv_monitor python=3.9
conda activate cctv_monitor
pip install -r requirements.txt

### 2. Set up the Environment

python cctv_monitor.py

