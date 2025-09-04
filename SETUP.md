# Environment Setup for Pyrite Development

## Prerequisites
- Python 3.8 or higher
- Git

## Setup Instructions

### 1. Clone the repository
```bash
git clone [repository-url]
cd Pyrite
```

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Activate virtual environment

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install exact dependencies
```bash
# Install from lock file for exact environment match
pip install -r requirements-lock.txt

# Install package in development mode
pip install -e .
```

### 5. Verify installation
```bash
python -c "import pyrite; print('Pyrite installed successfully!')"
```

## Current Package Versions (as of setup)
- Python 3.11
- numpy-2.3.2
- pandas-2.3.2  
- scipy-1.16.1
- pyrite-0.1.0

## Updating Dependencies

If you add new packages:
1. Install the package: `pip install package-name`
2. Update lock file: `pip freeze > requirements-lock.txt`
3. Commit and push: `git add requirements-lock.txt && git commit -m "Update dependencies" && git push`
4. Other developer pulls and reinstalls: `git pull && pip install -r requirements-lock.txt`