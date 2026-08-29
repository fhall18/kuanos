"""Training script for the weather prediction model."""

import numpy as np
import pandas as pd
import plotnine as p9
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time 
from joblib import Parallel, delayed
from bs4 import BeautifulSoup

import json
from pathlib import Path

import requests
from datetime import datetime
from datetime import datetime, timezone, timedelta

import lightgbm as lgb
import statsmodels.api as sm
from sklearn.metrics import (
    mean_absolute_percentage_error, mean_squared_error, r2_score,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import GridSearchCV, KFold