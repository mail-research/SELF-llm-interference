## Getting Started 🎯
### Installation
```bash
# Recommend Python 3.10.
pip install -r requirements.txt
pip install vllm==0.8.5.post1
cd verl
pip install -e .
```

### Data
Our raw training data in `deepscaler/data/[train|test]`, along with preprocessing scripts. To convert the raw data into Parquet files for training, run:
```python
# Output parquet files in data/*.parquet.
python scripts/data/deepscaler_dataset.py
```
To preprocess AIME25, run:
```python
python scripts/data/aime25.py
```
### Training Scripts

We provide training scripts for both single-node setup in `slurm`.
For GRPO Training:
```python
bash slurm train_grpo.sh
```
For SELF Training:
```python
bash slurm train_grpo.sh
```
