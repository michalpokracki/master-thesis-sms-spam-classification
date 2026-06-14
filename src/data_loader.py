"""
Data loading for the UCI SMS Spam Collection.

The dataset is distributed as a ZIP archive containing a tab-separated file
(`SMSSpamCollection`) with two columns: the label ("ham" or "spam") and the raw
message text. It is *not* exposed as a direct CSV import, so this module
downloads the official archive and parses it.

Terminology
-----------
The dataset uses two labels:
- "ham"  : a legitimate (non-spam) message — the term is the conventional
           counterpart of "spam" in the spam-filtering literature.
- "spam" : an unsolicited message.
"""

from __future__ import annotations

import io
import os
import ssl
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # certifi is listed in requirements; this is a safety net
    _SSL_CONTEXT = ssl.create_default_context()

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)

# Cache the parsed dataset locally so we do not re-download on every run.
_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_FILE = _CACHE_DIR / "sms_spam.csv"


def _download_and_parse() -> pd.DataFrame:
    """Download the UCI archive and parse the tab-separated collection."""
    request = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, context=_SSL_CONTEXT) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        # The collection file has no extension and is named "SMSSpamCollection".
        target = next(
            name for name in archive.namelist() if name.endswith("SMSSpamCollection")
        )
        with archive.open(target) as handle:
            df = pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=["label", "message"],
                encoding="latin-1",
            )
    return df


def load_sms_spam(use_cache: bool = True) -> pd.DataFrame:
    """
    Return the SMS Spam Collection as a DataFrame with columns:
    `label` ("ham"/"spam"), `message` (str), and `label_binary` (0 = ham, 1 = spam).

    Parameters
    ----------
    use_cache : bool
        If True, read from / write to a local CSV cache under ``data/``.
    """
    if use_cache and _CACHE_FILE.exists():
        df = pd.read_csv(_CACHE_FILE)
    else:
        df = _download_and_parse()
        if use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(_CACHE_FILE, index=False)

    df["label"] = df["label"].str.strip().str.lower()
    df["label_binary"] = (df["label"] == "spam").astype(int)
    return df


if __name__ == "__main__":
    data = load_sms_spam()
    print(f"Loaded {len(data)} messages")
    print(data["label"].value_counts())
    print(data.head())
