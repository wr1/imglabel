"""Save and load criteria."""

import yaml


def save_criteria(avg_hsv, threshold_hue, threshold_sat, filename="criteria.yaml"):
    """Save filter criteria to YAML."""
    criteria = {
        "hsv": avg_hsv.tolist(),
        "threshold_hue": threshold_hue,
        "threshold_sat": threshold_sat
    }
    with open(filename, "w") as f:
        yaml.dump(criteria, f)


def load_criteria(filename="criteria.yaml"):
    """Load filter criteria from YAML."""
    with open(filename, "r") as f:
        criteria = yaml.safe_load(f)
    return criteria["hsv"], criteria["threshold_hue"], criteria["threshold_sat"]
