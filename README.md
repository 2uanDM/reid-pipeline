# Person Re-Identification Optimized Pipeline
This repository represents the work for paper *"Person Re-Identification with Edge Devices: Effectively utilizing Computing Resource with Frame Skipping and Batch Processing mechanism"*

## 🚀 Dependencies Installation

1. Install `uv`
```bash
pip install uv
```

`uv` is a python framework manager like `pip`, but written in Rust. It is faster than `pip` and can be used to install python packages parallelly.

From now, whenever adding a new dependency, use `uv` to install it by just adding the package name to `requirements.txt` and run the following command.

2. Install dependencies
```bash
uv pip install -r requirements.txt
```

3. Docker services (`Kafka`)
```bash
docker compose up -d
```
