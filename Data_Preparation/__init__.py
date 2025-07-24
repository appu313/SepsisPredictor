"""
Data_Preparation package

This package defines the data preparation steps for the Deep Learning S25 Course Project.
Each module is owned by a specific team member for clear responsibility.

Modules and owners:
-------------------
- data_parsing.py               --> Responsible: Mitch (Data parsing & handling missing data)
- label_generation_split.py     --> Responsible: Aparna (Label generation & dataset splitting)
- feature_normalization.py      --> Responsible: Asal (Feature normalization & class imbalance handling)
- eda.py                        --> Responsible: Ehsan (Exploratory Data Analysis)
"""

# Import each step with clear function names and owner notes
from .data_parsing import parse_and_clean_data  # Mitch's part
from .label_generation_split import stratified_group_k_fold  # Asal's part
from .feature_normalization import (
    center,
    train_validate_split,
    smote_oversample_to_tensor,
)  # Aparna's part
from .eda import (
    run_eda,
    run_comprehensive_eda,
    corr_difference_analysis,
)  # Ehsan's part

from .grud_preprocessing import generate_grud_input

from .transformer_preprocessing import generate_transformer_input