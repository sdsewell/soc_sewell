## Known pytest issue (2026-05-13)

`conftest.py` contains a stale import:
    `from src.two_d_one_d_reduction.tolansky import ...`
This module no longer exists.  Until conftest.py is cleaned up, run the
Tolansky test suite with:
    pytest tests/test_tolansky_2026-05-13.py --noconftest -v
or fix conftest.py by removing the stale import line.
