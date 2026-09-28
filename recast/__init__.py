"""ReCAST reward credit assignment for diffusion fine-tuning."""

from .config import Recipe, WeightingConfig, load_recipe
from .weighting import (
    WeightTable,
    calculate_weight_table,
    load_weight_table,
    save_weight_table,
)

__all__ = [
    "Recipe",
    "WeightTable",
    "WeightingConfig",
    "calculate_weight_table",
    "load_recipe",
    "load_weight_table",
    "save_weight_table",
]

__version__ = "0.3.2"
