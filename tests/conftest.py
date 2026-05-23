import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simtrade.db import init_db  # noqa: E402


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "test.db"
    c = init_db(db)
    yield c
    c.close()
