# VeriScan — Complete Installation & Setup Commands Guide
Repository: https://github.com/Praveen-Suthar-08/Veriscan-ai-agent.git

---

## 1. System Prerequisites

Before running the Python commands, ensure the host machine has:
- **Git** (v2.25+)
- **Python 3.10, 3.11, or 3.12** (Python 3.11 recommended).
  *Note for Windows users: Check "Add Python to PATH" during installation.*
- *(Optional)* **Tesseract-OCR Engine**:
  - Required only if running live OCR on custom un-cached scans.
  - Bundled sample cases use pre-computed token caches, so the demo runs immediately even without Tesseract.

---

## 2. Setup Commands by Operating System

### 🪟 Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/Praveen-Suthar-08/Veriscan-ai-agent.git
cd Veriscan-ai-agent

# 2. Create a virtual environment
python -m venv venv

# 3. Allow script execution and activate virtual environment
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install all required dependencies
pip install -r requirements.txt

# 6. Verify installation by running test suite (113 tests)
pytest -q

# 7. Launch the VeriScan Streamlit application
streamlit run app/main.py
```

---

### 🪟 Windows (Command Prompt / CMD)

```cmd
:: 1. Clone repository
git clone https://github.com/Praveen-Suthar-08/Veriscan-ai-agent.git
cd Veriscan-ai-agent

:: 2. Create virtual environment
python -m venv venv

:: 3. Activate virtual environment
venv\Scripts\activate.bat

:: 4. Upgrade pip
python -m pip install --upgrade pip

:: 5. Install dependencies
pip install -r requirements.txt

:: 6. Run tests
pytest -q

:: 7. Launch application
streamlit run app/main.py
```

---

### 🍎 macOS & 🐧 Linux (Ubuntu / Debian / WSL)

```bash
# 1. (Optional for Ubuntu) Install system dependencies and Tesseract
sudo apt update && sudo apt install -y python3-pip python3-venv git tesseract-ocr

# (Optional for macOS via Homebrew)
# brew install git tesseract

# 2. Clone repository
git clone https://github.com/Praveen-Suthar-08/Veriscan-ai-agent.git
cd Veriscan-ai-agent

# 3. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run test suite
pytest -q

# 6. Launch application
streamlit run app/main.py
```

---

## 3. Makefile Shortcuts (If `make` is installed)

```bash
make setup      # Checks environment, installs dependencies, verifies Tesseract
make test       # Runs full pytest test suite (113 tests)
make eval       # Re-runs benchmark evaluation across 52 test cases
make data       # Regenerates synthetic dataset and ground truth
make run        # Runs Streamlit app directly
make demo       # Regenerates data and runs the app
```

---

## 4. Complete List of Dependencies Installed (`requirements.txt`)

- **Web UI & Reports**: `streamlit>=1.32.0`, `reportlab>=4.0.0`
- **Data & Models**: `pydantic>=2.0`, `PyYAML>=6.0`, `numpy>=1.24.0`, `pandas>=2.0.0`
- **Computer Vision & Image**: `opencv-python-headless>=4.8.0`, `Pillow>=10.0.0`, `PyMuPDF>=1.23.0`
- **OCR & Text Matching**: `pytesseract>=0.3.10`, `easyocr>=1.7.0`, `rapidfuzz>=3.0.0`, `jellyfish>=1.0.0`, `dateparser>=1.1.8`
- **Synthetic Data & Testing**: `Faker>=24.0.0`, `matplotlib>=3.8.0`, `pytest>=8.0.0`
- **Optional LLMs**: `google-generativeai>=0.4.0`, `anthropic>=0.18.0`, `requests>=2.31.0`

---

## 5. Optional: LLM Configuration

VeriScan runs **100% offline with zero API keys required** (`LLM_PROVIDER=template`).

To enable cloud LLM narrative summaries:
1. Copy `.env.example` to `.env`:
   - Linux/macOS: `cp .env.example .env`
   - Windows: `copy .env.example .env`
2. Add your provider key:
   - **Gemini**: `LLM_PROVIDER=gemini` + `GEMINI_API_KEY=your_key`
   - **Anthropic**: `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY=your_key`
   - **Ollama (Local)**: `LLM_PROVIDER=ollama` + `OLLAMA_URL=http://localhost:11434`

---

## 6. Accessing the Application

After running `streamlit run app/main.py`, open your browser to:
**http://localhost:8501**
