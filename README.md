# Person Re-Identification Optimized Pipeline
This repository represents the work for paper *"Real-Time Person Re-Identification and Tracking
on Edge Devices with Distributed Optimization"*

## 🚀 Dependencies Installation

*Note*: The repository is recommended to be installed in Linux/Unix environment

1. Install `uv`
```bash
pip install uv
```

2. Install dependencies
```bash
uv pip install -r requirements.txt
```

3. Docker services (`Kafka` & Redis)
```bash
docker compose up -d
```

## 📦 Dataset Preparation

1. Download the dataset and copy it to `src/assets` folder
```bash
make download
```

## 📦 Testing
1. Run the server
```bash
python main.py --server
```

2. Run the edge device on another terminal
```bash
python main.py --edge --cpu --source src/assets/videos/720p_60fps_35s.ts
```

*Note*: If you want to test the edge device on a separate machine similar to the paper, you need to change the `KAFKA_HOST` in `src/config.py` to the IP address of the machine running the server.