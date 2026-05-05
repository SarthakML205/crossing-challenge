# Dockerfile — my_experiments Commit 4: LightGBM Intent + Ego-Fusion GRU Trajectory
#
# Requires model.pkl (train_lgbm.py) and gru_model/ (train_gru.py) before building.
#
# Build:
#   docker build -t crossing-ego-gru .
#
# Test grader pathway (mount the data folder):
#   docker run --rm -v $(pwd)/../crossing-challenge-starter/data:/work \
#       crossing-ego-gru /work/dev.parquet /work/preds.csv
#
# Constraints:
#   Total image <= 2 GB  |  4 CPUs  |  4 GB RAM  |  --network=none

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Single RUN layer: system deps + PyTorch CPU wheel + remaining requirements.
# Torch is installed from the PyTorch index first so pip resolves the CPU-only
# wheel (~230 MB) instead of the CUDA-bundled package from PyPI (~750 MB+).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Submission surface: predict.py + grade.py + features.py + trained weights.
COPY predict.py grade.py features.py ./
COPY model.pkl ./
COPY gru_model/ ./gru_model/

# Grader invokes:  python grade.py <input.parquet> <output.csv>
ENTRYPOINT ["python", "grade.py"]
