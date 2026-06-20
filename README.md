# PyTorch Project

A PyTorch project configured for Google Colab and local development.

## Quick Start in Google Colab

Click the button below to open this project in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Piyush-08-bot/pytorch/blob/main/notebook.ipynb)

## Setup Instructions

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab

The notebook (`notebook.ipynb`) includes setup cells that will:

- Mount Google Drive (optional)
- Install required packages
- Download datasets if needed

## Project Structure

```
├── notebook.ipynb          # Main Colab notebook
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── model.py           # Model definitions
│   ├── train.py           # Training logic
│   └── utils.py           # Utility functions
├── data/                  # Data directory
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full list

## Development Tips

- Use `src.utils.set_seed()` for reproducible results
- Call `src.utils.count_parameters()` to inspect model size
- Check device availability with `src.utils.get_device()`
