# VeriScan Makefile — National AI Hackathon 2026, Track C (Problem C2)

.PHONY: setup data test eval run demo clean

setup:
	python -m pip install -r requirements.txt
	python -c "import shutil; print('[INFO] Tesseract binary status:', 'FOUND at ' + shutil.which('tesseract') if shutil.which('tesseract') else 'NOT in PATH (Optional: Pre-cached tokens and EasyOCR run without Tesseract binary)')"
	python -c "import veriscan; print('[SUCCESS] VeriScan package imported successfully. Version:', veriscan.__version__)"

data:
	python data/generate_mock.py

test:
	pytest tests/ -v

eval:
	python eval/run_eval.py

run:
	streamlit run app/main.py

demo: data
	streamlit run app/main.py

clean:
	python -c "import shutil, os, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]"

cache:
	python veriscan/data/cache_demo_tokens.py

