# ArtGAN-GP

WGAN-GP for abstract art generation, trained from scratch on a WikiArt subset (Abstract Expressionism / Color Field), 64x64.

Course project. Grading: 85% Baseline (code quality, ambition) + 15% Extra Criteria (Scaled MLOps + Hyperparameter tuning).

## Quick start

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login

python -m src.training.train --config-name=baseline
```

### Docker (GPU)

```bash
export WANDB_API_KEY=<your-key>
docker compose -f docker/docker-compose.yml up train
```

### Sweep

```bash
python scripts/run_sweep.py
```

### Tests

```bash
pytest tests/
ruff check . && ruff format .
```

## Layout

```
src/
  data/         # WikiArt loader, transforms
  models/       # Generator, Critic
  training/     # train loop, gradient penalty
  utils/        # logging, checkpoints, image grids
  config/       # Hydra YAML configs
scripts/        # download_data.py, run_sweep.py
docker/         # Dockerfile, docker-compose.yml
tests/          # unit tests (shapes, GP correctness)
notebooks/      # EDA, latent space interpolation
```
