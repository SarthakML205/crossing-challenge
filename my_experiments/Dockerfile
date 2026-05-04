# Dockerfile — my_experiments Commit 1: Constant Velocity Baseline
#
# No trained model weights needed: trajectory uses constant velocity,
# intent uses a fixed 0.5 prior.
#
# Build:
#   docker build -t crossing-cv .
#
# Test grader pathway (mount the data folder):
#   docker run --rm -v $(pwd)/../crossing-challenge-starter/data:/work \
#       crossing-cv /work/dev.parquet /work/preds.csv
#
# Run smoke test locally before building:
#   python tests/smoke.py
#
# Constraints:
#   Total image <= 2 GB  |  4 CPUs  |  4 GB RAM  |  --network=none

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# No model.pkl — this submission is purely algorithmic.
COPY predict.py grade.py ./

# Grader invokes:  python grade.py <input.parquet> <output.csv>
ENTRYPOINT ["python", "grade.py"]
