# Dockerfile — my_experiments Commit 3: LightGBM Intent + GRU Trajectory Predictor
#
# Requires model.pkl (train_lgbm.py) and gru_model/ (train_gru.py) before building.
#
# Build:
#   docker build -t crossing-gru .
#
# Test grader pathway (mount the data folder):
#   docker run --rm -v $(pwd)/../crossing-challenge-starter/data:/work \
#       crossing-gru /work/dev.parquet /work/preds.csv
#
# Run smoke test locally before building:
#   python tests/smoke.py
#
# Constraints:
#   Total image <= 2 GB  |  4 CPUs  |  4 GB RAM  |  --network=none

FROM python:3.11-slim

WORKDIR /app

# libgomp1 required by LightGBM on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install PyTorch CPU-only wheel first (avoids pulling CUDA deps via PyPI)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Submission surface: predict.py + grade.py + features.py + trained weights.
COPY predict.py grade.py features.py ./
COPY model.pkl ./
COPY gru_model/ ./gru_model/

# Grader invokes:  python grade.py <input.parquet> <output.csv>
ENTRYPOINT ["python", "grade.py"]
