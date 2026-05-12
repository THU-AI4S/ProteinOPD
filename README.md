# ProteinOPD

ProteinOPD currently contains two training and generation tracks:

- `unconditional`: ProtGPT2 prefix-tuning teacher construction, ProtGPT2 geometric ProteinOPD training, and unconditional generation.
- `conditional`: ProLLaMA LoRA instruction-tuning teacher construction, ProLLaMA geometric ProteinOPD training, and conditional generation.

## Environment Setup

The repository provides two dependency files:

- `environment.yml`: create a fresh conda environment with PyTorch/CUDA and the project dependencies.
- `requirements.txt`: install the project Python dependencies into an existing PyTorch/CUDA environment. It does not install `torch`.

Option 1: create a fresh conda environment.

```bash
conda env create -f environment.yml
conda activate proteinopd
```

`environment.yml` uses `pytorch-cuda=12.1` by default. If your server driver or cluster image requires another CUDA version, edit the `pytorch-cuda` version in `environment.yml` before creating the environment.

Option 2: install the project dependencies into an existing PyTorch/CUDA conda environment.

```bash
conda activate <your_env>
pip install -r requirements.txt
```

`requirements.txt` does not install PyTorch to avoid replacing a CUDA-matched cluster installation. Install PyTorch using the method recommended for your cluster, or use `environment.yml`.

Check the core dependencies after installation:

```bash
python -c "import torch, transformers, datasets, peft, accelerate, yaml; print('torch', torch.__version__)"
```

## ProtGPT2 Prefix Teacher Construction

Edit the configuration file:

```text
unconditional/teacher_construct/prefix_tuning_prot.yaml
```

Run:

```bash
python unconditional/teacher_construct/prefix_tuning_prot.py \
  --config unconditional/teacher_construct/prefix_tuning_prot.yaml
```

The input CSV files must contain the sequence column specified by `text_column`; the default is `sequence`. The output is a PEFT prefix adapter.

## ProLLaMA LoRA Teacher Construction

Edit the model, data, and output paths in:

```text
conditional/teacher_construct/scripts/run_it_lora_multigpu.sh
```

Run:

```bash
bash conditional/teacher_construct/scripts/run_it_lora_multigpu.sh
```

The training data should be instruction JSON/JSONL with `instruction`, `input`, and `output` fields. The output is a ProLLaMA LoRA adapter.

## ProtGPT2 Geometric ProteinOPD

Edit the teacher configuration:

```text
unconditional/proteinopd/configs/teachers.yaml
```

Run:

```bash
bash unconditional/proteinopd/scripts/run_protein_opd_ddp.sh
```

## ProLLaMA Geometric ProteinOPD

Edit the configuration file:

```text
conditional/proteinopd/configs/prollama_protein_opd_example.yaml
```

Run:

```bash
bash conditional/proteinopd/scripts/run_prollama_protein_opd_ddp.sh
```

## Generation

ProtGPT2 unconditional generation:

```bash
python unconditional/generate/generate.py \
  --config unconditional/generate/generate.yaml
```

ProLLaMA conditional generation:

```bash
python conditional/generate/generate.py \
  --config conditional/generate/generate.yaml
```

Both `generate.py` scripts read `generate.yaml` from the same directory by default.
