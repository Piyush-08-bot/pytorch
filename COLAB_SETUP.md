# Google Colab Setup Guide

This guide explains how to use this PyTorch project in Google Colab.

## Quick Start

1. **Open in Colab**: Click the "Open In Colab" badge in the README.md
2. **Run Setup Cells**: Execute the setup cells to configure the environment
3. **Start Coding**: Use the main notebook cells or add your own

## Step-by-Step Setup

### 1. Configure GPU (Recommended)

- Click: **Runtime** → **Change runtime type**
- Select: **GPU** (T4 or better)
- Click: **Save**

### 2. Install Dependencies

Run this cell to install all required packages:

```python
%pip install torch torchvision numpy pandas matplotlib scikit-learn jupyter
```

### 3. Mount Google Drive (Optional but Recommended)

To save your work persistently:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then save files to: `/content/drive/MyDrive/`

### 4. Clone Your Repository

If your code is on GitHub:

```bash
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

### 5. Import Your Project

```python
import sys
sys.path.insert(0, './src')

from model import SimpleNN
from utils import get_device
```

## Working with the Notebook

### Tips & Tricks

- **Install packages**: Use `%pip install package_name`
- **Run shell commands**: Use `!command`
- **Check GPU**: `!nvidia-smi`
- **Upload files**: `from google.colab import files; files.upload()`
- **Download files**: `files.download('model.pth')`

### Saving Your Work

**Option 1: Save to Google Drive**

```python
import shutil
shutil.copy('model.pth', '/content/drive/MyDrive/model.pth')
```

**Option 2: Download to Local**

```python
from google.colab import files
files.download('model.pth')
```

## Common Issues & Solutions

### Issue: Module not found

**Solution**: Make sure to add the project path to sys.path:

```python
import sys
sys.path.insert(0, './src')
```

### Issue: GPU not available

**Solution**:

1. Go to Runtime → Change runtime type
2. Select GPU
3. Restart the runtime

### Issue: Out of Memory (OOM)

**Solution**:

- Reduce batch size
- Reduce model size
- Use gradient checkpointing
- Enable mixed precision training

## Data Management

### Upload Data

```python
from google.colab import files
uploaded = files.upload()
```

### Download Data

```python
import shutil
shutil.make_archive('data', 'zip', 'data')
files.download('data.zip')
```

## Deployment Tips

1. **Save model checkpoint**:

   ```python
   torch.save(model.state_dict(), 'model.pth')
   ```

2. **Save entire model**:

   ```python
   torch.save(model, 'model_complete.pth')
   ```

3. **Save training artifacts**:
   ```python
   import json
   with open('metrics.json', 'w') as f:
       json.dump(metrics, f)
   ```

## References

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [CUDA with PyTorch](https://pytorch.org/get-started/locally/)
