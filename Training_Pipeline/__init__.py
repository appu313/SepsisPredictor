"""
Training_Pipeline package

This package defines the training pipeline for the Deep Learning S25 Course Project.
Each module is owned by a specific team member for clear responsibility.

Modules and owners:
-------------------
- train.py            --> Responsible: Mitch (Training loop and DataLoader initialization)
- tune.py             --> Responsible: Mitch (Grid search hyperparameter tuning)
- evaluate.py         --> Responsible: Mitch (Implement evaluation metrics (AUROC, PRC, etc.))
- visualize.py        --> Responsible: _____ (Logging & experiment tracking (TensorBoard))
"""
from .train import(
    train_validate,
    train_for_evaluation, 
    process_train_task,
    Train_Hyperparameter_Grid, 
    Train_Hyperparameters,
    Train_Val_Task
)
from .tune import grid_search_tune, grid_search_tune_parallel, Hyperparameter_Grid
from .evaluate import roc_eval