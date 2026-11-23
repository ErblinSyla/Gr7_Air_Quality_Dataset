"""Exploratory data analysis for the Phase I stratified hourly sample.

Run after Phase I has produced data/gold/sample_10pct.parquet. This script is
read-only with respect to the input and writes tables, figures, and a report
under reports/eda/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent
POLLUTANTS = ["SO2", "PM10", "PM2_5", "O3", "NO2", "NOx", "CO", "NO"]