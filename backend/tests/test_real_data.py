"""Checks the REAL data under DATA_DIR. Opt-in; run on the server:

    cd ~/ieee-ai-demo/backend && REAL_DATA=1 python -m pytest tests/test_real_data.py

Uses mmap and chunked reads, so it is safe on the 16 GB CPU plan. Takes a few minutes
(the quickdraw JSONL lines are counted; only the first 20,000 per split are fully parsed).
"""

import os

import pytest

from app.config import get_settings
from app.data import contract

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(os.environ.get("REAL_DATA") != "1", reason="set REAL_DATA=1 to check the real DATA_DIR"),
]


@pytest.fixture(scope="module")
def data_dir():
    return get_settings().data_dir


def test_real_quickdraw(data_dir):
    assert contract.check_quickdraw_processed(data_dir / "quickdraw" / "processed", real=True, max_records=20_000) == []


def test_real_speech(data_dir):
    assert contract.check_speech_processed(data_dir / "speech" / "processed", real=True) == []


def test_real_lichess(data_dir):
    assert contract.check_lichess(data_dir / "lichess", real=True) == []
