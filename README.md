# PharmMLPK MVP

Pharmacokinetics modeling with PyTorch and graph neural networks (GNN). This repository provides a clean starter layout, virtual environment workflow, and placeholder training code.

## Prerequisites

- **Python 3.10+** (tested with Python 3.10 on macOS)
- Terminal access from the project root (`PharmMLPK_MVP/`)

Check Python:

```bash
python3 --version
```

## 1. Create the virtual environment

Run from the **project root** (the folder that contains `src/`, `data/`, and `requirements.txt`):

```bash
cd /path/to/PharmMLPK_MVP
python3 -m venv .venv
```

## 2. Activate the environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt should show `(.venv)` when the environment is active.

## 3. Upgrade pip and install dependencies

Still in the project root with `.venv` activated:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Optional: PyTorch Geometric (GNN)

`torch-geometric`, `torch-scatter`, and `torch-sparse` depend on your **exact PyTorch and CUDA/CPU build**. If `pip install -r requirements.txt` fails on those lines, install PyTorch first, then follow the [official PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html):

```bash
# After torch is installed — use the wheel index matching your torch version
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
```

Replace the wheel URL with the combination listed for your `torch` version on the PyG website.

## 4. Test the setup

From the project root with `.venv` activated:

```bash
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"   # macOS/Linux
python tests/test_environment.py
```

On Windows (PowerShell):

```powershell
$env:PYTHONPATH = "$PWD"
python tests/test_environment.py
```

You should see package versions and `Environment check PASSED.`

## 5. Run starter scripts

**GNN placeholder forward pass:**

```bash
python -m src.models.gnn_model
```

**Dummy training loop:**

```bash
python -m src.training.train
```

**Data loader helper:**

```bash
python -m src.data.load_data
```

## Project layout

```
PharmMLPK_MVP/
├── data/
│   ├── raw/          # place raw CSVs here (gitignored contents)
│   └── processed/
├── src/
│   ├── data/         # load_data.py
│   ├── models/       # gnn_model.py
│   ├── training/     # train.py
│   └── utils/        # config.py
├── tests/
│   └── test_environment.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Regenerating `requirements.txt`

After adding packages inside `.venv`:

```bash
pip freeze > requirements.txt
```

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| `ModuleNotFoundError: src` | Set `PYTHONPATH` to the project root or use `python -m src....` |
| PyG install fails | Install matching wheels from [pyg.org](https://data.pyg.org/whl/) for your torch version |
| MPS on Apple Silicon | PyTorch uses `mps` when available; training config uses `device="auto"` |

## License

Add your license here.
