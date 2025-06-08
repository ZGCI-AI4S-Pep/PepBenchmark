Environment Setup

Install the required dependencies in a new plm conda environment:

# Create and activate the conda environment
conda create -n plm python=3.10 -y
conda activate plm

# GPU support
conda install cudatoolkit=12.1 -y
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Model and utility libraries
conda install transformers==4.52.4 -y
conda install scikit-learn==1.7.0 -y
conda install accelerate==1.7.0 -y
conda install wandb==0.20.1 -y
conda install rdkit=2024.03.6 -y
conda install optuna=4.3.0 -y
conda install ipykernel=6.29.5 -y

This will install:

PyTorch 2.5.1 with CUDA 12.1 support

Hugging Face Transformers

Scikit-learn, Accelerate, Weights & Biases, RDKit

Optuna for hyperparameter optimization

IPython kernel

Single Experiment Run

Run one fine‑tuning job with specified parameters:
```bash
WANDB_PROJECT=test python finetune_plm.py \
    --task AF_APML \
    --model_name esm2_150M \
    --num_train_epochs 30 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --early_stopping_patience 5 \
    --weight_decay 0.0 \
    --split_type Random_split \
    --split_index random1 \
    --output_dir test
```

Results will be logged to the Weights & Biases project test and saved under `exp_root = Path(args.output_dir or Path("checkpoints") / args.task / args.split_type / args.model_name)`

Hyperparameter Tuning

Use Optuna to automatically search for the best hyperparameters:

WANDB_PROJECT=test python finetune_plm_optuna.py \
  --task AF_APML \                      # Dataset name
  --model_name dplm_150m \              # Pretrained model identifier
  --per_device_train_batch_size 16 \    # Fixed batch size per device
  --early_stopping_patience 5 \         # Early stopping patience
  --weight_decay 0.0 \                  # Fixed weight decay
  --split_type Homology_based_split \   # Data split method
  --split_index random1 \               # Split index
  --auto_tune \                         # Enable automatic tuning
  --n_trials 10 \                       # Number of Optuna trials
  --output_dir test                     # Output directory

The best hyperparameters and tuned model will be stored in the tuning/ subdirectory of the experiment folder.
`exp_root = Path(args.output_dir or Path("checkpoints") / args.task / args.split_type / args.model_name)`
Batch Experiment Scripts

Run all experiments for a single dataset:

chmod +x run_experiments
./run_experiments

This script iterates over tasks, models, split types, and split indices to execute all fine‑tuning runs.

Run experiments for all datasets:

chmod +x run_all
./run_all

This script loops through every dataset, model, and split configuration to perform a complete set of training and evaluation tasks.

License

Specify your project’s license here.

