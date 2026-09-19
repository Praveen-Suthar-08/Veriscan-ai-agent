"""Automated test asserting sample watermark and synthetic data compliance."""

import json
from pathlib import Path
from veriscan.data.generate_mock import SAMPLES_DIR, DEV_DIR, HELDOUT_DIR, WATERMARK_TEXT



class TestWatermarkAndSyntheticData:
    """Verify that all generated samples contain the synthetic watermark and ground truth."""

    def test_watermark_constant(self):
        assert WATERMARK_TEXT == "SAMPLE – NOT A REAL DOCUMENT"

    def test_bundled_samples_watermarked(self):
        sample_cases = list(SAMPLES_DIR.iterdir())
        assert len(sample_cases) >= 4

        for cdir in sample_cases:
            if not cdir.is_dir():
                continue
            truth_path = cdir / "truth.json"
            assert truth_path.exists(), f"truth.json missing in {cdir}"
            with open(truth_path, "r", encoding="utf-8") as f:
                truth = json.load(f)

            assert truth.get("watermark") == WATERMARK_TEXT
            for doc_key, dinfo in truth["documents"].items():
                img_path = cdir / dinfo["filename"]
                assert img_path.exists(), f"Image {img_path} missing"
                assert img_path.stat().st_size > 1000, f"Image {img_path} is empty"

    def test_dev_samples_watermarked(self):
        dev_cases = list(DEV_DIR.iterdir())
        assert len(dev_cases) >= 40

        for cdir in dev_cases:
            if not cdir.is_dir():
                continue
            truth_path = cdir / "truth.json"
            assert truth_path.exists()
            with open(truth_path, "r", encoding="utf-8") as f:
                truth = json.load(f)
            assert truth.get("watermark") == WATERMARK_TEXT

    def test_heldout_samples_watermarked(self):
        heldout_cases = list(HELDOUT_DIR.iterdir())
        assert len(heldout_cases) >= 12

        for cdir in heldout_cases:
            if not cdir.is_dir():
                continue
            truth_path = cdir / "truth.json"
            assert truth_path.exists()
            with open(truth_path, "r", encoding="utf-8") as f:
                truth = json.load(f)
            assert truth.get("watermark") == WATERMARK_TEXT
