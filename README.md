# ArtGAN-GP

WGAN-GP for abstract art generation, trained from scratch on a WikiArt subset (Abstract Expressionism / Color Field), 64x64.

Course project. Grading: 85% Baseline (code quality, ambition) + 15% Extra Criteria (Scaled MLOps + Hyperparameter tuning).

## What's inside

- **WGAN-GP from scratch** — Generator + Critic with the load-bearing WGAN-GP invariants enforced (no BatchNorm in the Critic, Adam betas (0, 0.9), n_critic = 5, gradient penalty with per-sample epsilon).
- **Hydra configs** — every dial (model, data, training, sweep) is in `src/config/`.
- **W&B integration** — runs, image grids, checkpoint and dataset Artifacts, FID metric.
- **Docker** — CUDA PyTorch base image with GPU passthrough through `docker compose`.
- **Tests** — unit tests for the gradient penalty (closed-form), Critic-has-no-BatchNorm invariant, and an end-to-end loop smoke test.

## End-to-end reproduction

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login
```

Or with Docker:

```bash
export WANDB_API_KEY=<your-key>
make docker-build
```

### 2. Data

Stream the WikiArt abstract subset from Hugging Face, center-crop and resize to 64x64, and save PNGs into `data/wikiart_abstract/`:

```bash
python -m scripts.download_data
```

Optional: publish the preprocessed dataset as a W&B Artifact so sweep workers can pull an identical snapshot.

```bash
python -m scripts.download_data data.upload_artifact=true
```

For a quick smoke run without committing to the full download:

```bash
python -m scripts.download_data data.max_images=200
```

### 3. Train

```bash
python -m src.training.train                                    # full run, W&B online
python -m src.training.train wandb.mode=disabled                # local smoke, no W&B traffic
python -m src.training.train training.epochs=10 data.batch_size=32
python -m src.training.train training.resume_from=checkpoints/epoch_0050.pt
```

Or in Docker:

```bash
make docker-train
```

Per-epoch behaviour:

- Logs critic loss, generator loss, **Wasserstein estimate**, gradient penalty, and critic gradient norm.
- Saves an 8x8 image grid generated from a fixed latent batch (so per-epoch progress is visually comparable).
- Checkpoints every `training.checkpoint_every_n_epochs` and logs them as W&B Artifacts.
- Computes FID every `training.fid_every_n_epochs` over `training.fid_num_samples` images.

### 4. Resume

```bash
python -m src.training.train training.resume_from=checkpoints/epoch_0050.pt
# or from a W&B artifact:
python -m src.training.train training.resume_from=entity/artgan-gp/wgan_gp_checkpoint:latest
```

### 5. Sweep (Phase 5)

```bash
python scripts/run_sweep.py
```

### 6. Tests + lint

```bash
make test
make lint
```

## Project layout

```
src/
  data/         # WikiArt loader, transforms
  models/       # Generator, Critic, weight init
  training/     # train loop, gradient penalty
  utils/        # logger, checkpoints, image grids, metrics (FID)
  config/       # Hydra YAML configs (data / model / training / wandb / sweep)
scripts/        # download_data.py, run_sweep.py
docker/         # Dockerfile, docker-compose.yml
tests/          # unit tests (shapes, GP correctness, training smoke)
notebooks/      # EDA, latent space interpolation (Phase 6)
```
