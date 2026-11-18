"""Streamlit UI smoke tests (headless, via streamlit.testing.AppTest)."""
import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

PAGES = ["views/home.py", "views/dashboard.py", "views/reconstruction.py", "views/comparison.py",
         "views/analytics.py", "views/model_info.py", "views/about.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=120)
    at.run()
    assert not at.exception, at.exception


def test_app_entrypoint():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception


def test_reconstruction_run_button_on_sample():
    at = AppTest.from_file(str(ROOT / "views/reconstruction.py"), default_timeout=180)
    at.run()
    run = [b for b in at.button if b.label == "Run Reconstruction"][0]
    run.click().run()
    assert not at.exception, at.exception
    assert at.session_state["result"] is not None
    assert at.session_state["result"]["metrics"]["RMSE"] > 0
    assert len(at.session_state["runs"]) == 1


def test_comparison_run():
    at = AppTest.from_file(str(ROOT / "views/comparison.py"), default_timeout=300)
    at.run()
    [b for b in at.button if b.label == "Run comparison"][0].click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["comparison"]["rows"]) == 7
