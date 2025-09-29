## Getting Started 🎯
### Installation
```bash
# Recommend Python 3.10.
pip install -r requirements.txt
cd verl
pip install -e .
```

### Data
Our raw training data in `deepscaler/data/[train|test]`, along with preprocessing scripts. To convert the raw data into Parquet files for training, run:
```python
# Output parquet files in data/*.parquet.
python scripts/data/deepscaler_dataset.py
```
### Training Scripts

We provide training scripts for both single-node and multi-node setups in `slurm/deepscaler`.
