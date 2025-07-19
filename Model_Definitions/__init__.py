"""
Model_Definitions package

This package defines the model definitions for the Deep Learning S25 Course Project.
Each module is owned by a specific team member for clear responsibility.

Modules and owners:
-------------------
- baselines.py          --> Responsible: Mitch (Baseline LSTM and GRU models)
- tcn.py                --> Responsible:
- gru_d.py              --> Responsible:
- transformer.py        --> Responsible:
- tft.py                --> Responsible:
"""

from .baselines import (
    Baseline_GRU,
    Baseline_LSTM,
    Baseline_Model_Hyperparameter_Grid,
    Basline_Model_Hyperparameters,
)

from .transformers import (
    Sepsis_Predictor_Encoder,
    Sepsis_Predictor_Encoder_Hyperparameters,
    PositionalEncoding,
    Dense_Interpolator
)

