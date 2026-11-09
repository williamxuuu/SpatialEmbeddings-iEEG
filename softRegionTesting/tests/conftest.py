"""Use the shared code and the fake Brain Treebank folder from spatialResolutionTesting."""
import os
import sys

SRT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "spatialResolutionTesting"))
sys.path[:0] = [os.path.join(SRT, "src"), os.path.join(SRT, "tests")]
os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/nonexistent")  # neuroprobe reads it at import

import pytest  # noqa: E402

try:  # the fake data folder needs neuroprobe + h5py; tests that use it skip otherwise
    from test_neuroprobe_adapter import fake_root  # noqa: E402,F401
except (ImportError, pytest.skip.Exception):  # importorskip raises Skipped at import
    pass
