"""
schema.py - Single source of truth for the corrosion-prediction system.

Built from the primary dataset (data/raw/corrosion_10000.xlsx, sheet
"Donnees_Corrosion"): 10,000 rows of physics-coherent synthetic data with
13 operating/chemistry features and a measured corrosion-rate target.

The data loader, the training pipeline, and the Streamlit app all import from
here, so column names, units, form fields, and model inputs never drift apart.
"""

import re
import unicodedata
from dataclasses import dataclass


# Raw French/Excel headers -> clean ASCII keys used everywhere in code.
COLUMN_MAP = {
    "ID": "id",
    "Matériau": "material",
    "Fluide": "fluid",
    "Phase": "phase",
    "Température (°C)": "temperature",
    "Pression (bar)": "pressure",
    "Densité (kg/m³)": "density",
    "Vitesse écoulement (m/s)": "flow_velocity",
    "Composition chimique": "chemistry",
    "pH": "ph",
    "Cl⁻ (mg/L)": "chloride",
    "CO₂ (%)": "co2",
    "H₂S (ppm)": "h2s",
    "O₂ (mg/L)": "o2",
    "Durée exposition (h)": "exposure_h",
    "Vitesse corrosion (mm/an)": "corrosion_rate",
    "Perte épaisseur (mm/an)": "thickness_loss_rate",
    "Remarques": "mechanism",
    "Norme / Référence": "reference",
}

GROUPS = ["Asset", "Fluid", "Operating conditions", "Fluid chemistry"]


@dataclass(frozen=True)
class NumericFeature:
    key: str
    label_en: str
    label_fr: str
    unit: str
    min: float
    max: float
    default: float
    group: str
    kind: str = "numeric"


@dataclass(frozen=True)
class CategoricalFeature:
    key: str
    label_en: str
    label_fr: str
    group: str
    # options are filled at runtime from models/metadata.json (derived from data)
    kind: str = "categorical"


# Ranges below are display hints for the form, taken from the dataset describe().
NUMERIC_FEATURES = [
    NumericFeature("temperature", "Temperature", "Température", "°C", 5, 210, 100, "Operating conditions"),
    NumericFeature("pressure", "Pressure", "Pression", "bar", 1, 350, 100, "Operating conditions"),
    NumericFeature("density", "Density", "Densité", "kg/m³", 0.7, 1236, 850, "Operating conditions"),
    NumericFeature("flow_velocity", "Flow velocity", "Vitesse d'écoulement", "m/s", 0.05, 12, 3.0, "Operating conditions"),
    NumericFeature("exposure_h", "Exposure duration", "Durée d'exposition", "h", 168, 17520, 1080, "Operating conditions"),
    NumericFeature("ph", "pH", "pH", "", 0.5, 14, 6.0, "Fluid chemistry"),
    NumericFeature("chloride", "Chloride (Cl⁻)", "Chlorures (Cl⁻)", "mg/L", 0, 180000, 300, "Fluid chemistry"),
    NumericFeature("co2", "CO₂", "CO₂", "%", 0, 100, 0.2, "Fluid chemistry"),
    NumericFeature("h2s", "H₂S", "H₂S", "ppm", 0, 6000, 0, "Fluid chemistry"),
    NumericFeature("o2", "Dissolved O₂", "O₂ dissous", "mg/L", 0, 14, 0.1, "Fluid chemistry"),
]

CATEGORICAL_FEATURES = [
    CategoricalFeature("material", "Material", "Matériau", "Asset"),
    CategoricalFeature("fluid", "Fluid", "Fluide", "Fluid"),
    CategoricalFeature("phase", "Phase", "Phase", "Fluid"),
    CategoricalFeature("chemistry", "Chemical composition", "Composition chimique", "Fluid"),
]

NUMERIC_KEYS = [f.key for f in NUMERIC_FEATURES]
CATEGORICAL_KEYS = [f.key for f in CATEGORICAL_FEATURES]
FEATURE_KEYS = NUMERIC_KEYS + CATEGORICAL_KEYS

# Identifiers / metadata / leakage / targets - never fed to the model as inputs.
DROP_FROM_FEATURES = ["id", "reference", "mechanism", "corrosion_rate", "thickness_loss_rate"]


@dataclass(frozen=True)
class Target:
    key: str
    label_en: str
    unit: str
    task: str


TARGETS = [
    Target("corrosion_rate", "Corrosion rate", "mm/year", "regression"),
    Target("thickness_loss_rate", "Thickness-loss rate", "mm/year", "regression"),
    Target("mechanism", "Likely corrosion mechanism", "", "classification"),
]

# NACE RP0775 corrosion-rate categories for carbon steel (mm/year).
RISK_BINS = [
    ("low", 0.0, 0.025),
    ("moderate", 0.025, 0.125),
    ("high", 0.125, 0.25),
    ("severe", 0.25, float("inf")),
]


def risk_from_rate(rate_mm_per_year: float) -> str:
    """Map a corrosion rate (mm/year) to a NACE RP0775 risk class."""
    for name, lo, hi in RISK_BINS:
        if lo <= rate_mm_per_year < hi:
            return name
    return "severe"


def rul_years(rate_mm_per_year: float, wall_thickness_mm: float, min_allowable_mm: float = 0.0) -> float:
    """Remaining useful life (years) = usable wall thickness / corrosion rate."""
    if not rate_mm_per_year or rate_mm_per_year <= 0:
        return float("inf")
    usable = max(wall_thickness_mm - min_allowable_mm, 0.0)
    return usable / rate_mm_per_year


def features_in_group(group: str):
    return [f for f in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if f.group == group]


# --------------------------------------------------------------------------- #
# Column-name reconciliation for uploaded datasets.
#
# Companies export data with their own headers - in French, English, the
# original Excel labels, or abbreviations. To feed any of them to the models we
# must map their columns onto the canonical FEATURE_KEYS. We normalise headers
# (drop "(unit)" suffixes, fold accents, turn ² / ₂ into 2, keep only
# alphanumerics) and match against a rich alias table built from each feature's
# key, its English + French labels, the original Excel header, and the common
# synonyms below.
# --------------------------------------------------------------------------- #

# Extra human aliases per canonical feature, on top of the key, the EN/FR
# labels and the original Excel headers (those are added automatically).
COLUMN_ALIASES = {
    "temperature": ["temp", "temperature c", "service temperature"],
    "pressure": ["press", "pressure bar", "operating pressure"],
    "density": ["rho", "masse volumique", "specific gravity", "fluid density"],
    "flow_velocity": ["flow", "velocity", "vitesse", "flow rate", "flow speed",
                      "fluid velocity", "debit"],
    "exposure_h": ["exposure", "exposure time", "exposure hours", "duration",
                   "duree", "service time", "duree exposition"],
    "ph": ["acidity", "p h"],
    "chloride": ["cl", "chlorures", "chlorides", "chlorure", "salinity",
                 "chloride content"],
    "co2": ["carbon dioxide", "co2 content", "pco2", "co2 partial pressure"],
    "h2s": ["hydrogen sulfide", "sour", "h2s content", "ph2s"],
    "o2": ["oxygen", "dissolved oxygen", "dissolved o2", "o2 content"],
    "material": ["matiere", "alloy", "steel", "steel grade", "grade",
                 "metallurgy", "material grade"],
    "fluid": ["medium", "service fluid", "product", "fluid type", "service"],
    "phase": ["state", "phase state", "flow phase"],
    "chemistry": ["chemical composition", "composition", "water chemistry",
                  "chem", "environment"],
}

# Optional wall-thickness columns the batch scorer also recognises (for RUL).
RUL_ALIASES = {
    "wall_thickness_mm": ["wall thickness", "wall", "thickness", "epaisseur",
                          "epaisseur paroi", "nominal thickness",
                          "current thickness", "wt"],
    "min_allowable_mm": ["min allowable", "minimum allowable", "min wall",
                         "epaisseur minimale admissible", "minimum thickness",
                         "retirement thickness", "t min"],
}


def normalize_header(s) -> str:
    """Fold a column header to a comparison key: no accents, units, or symbols.

    'Température (°C)' -> 'temperature', 'Cl⁻ (mg/L)' -> 'cl', 'CO₂ (%)' -> 'co2'.
    """
    s = re.sub(r"\(.*?\)", " ", str(s))                 # drop "(unit)" suffixes
    s = unicodedata.normalize("NFKD", s)                # ² -> 2, é -> e + accent
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _alias_lookup(include_rul: bool = True) -> dict:
    """Build {normalized_alias: canonical_key} from keys, labels, headers, aliases."""
    lookup: dict = {}

    def add(alias, key):
        n = normalize_header(alias)
        if n:
            lookup.setdefault(n, key)

    for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        add(f.key, f.key)
        add(f.label_en, f.key)
        add(f.label_fr, f.key)
    for raw, key in COLUMN_MAP.items():
        if key in FEATURE_KEYS:
            add(raw, key)
    for key, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            add(a, key)
    if include_rul:
        for key, aliases in RUL_ALIASES.items():
            add(key, key)
            for a in aliases:
                add(a, key)
    return lookup


def guess_column_mapping(user_columns, include_rul: bool = True) -> dict:
    """Best-guess {canonical_key: user_column or None} for an uploaded dataset.

    Matching is case-, accent- and unit-insensitive and recognises English,
    French, the original Excel headers and common abbreviations, so companies
    can keep their own column names. Each user column is used at most once;
    canonical features with no confident match map to None for the user to fix.
    """
    lookup = _alias_lookup(include_rul)
    norm_user: dict = {}
    for c in user_columns:
        norm_user.setdefault(normalize_header(c), c)    # first column wins

    targets = list(FEATURE_KEYS) + (list(RUL_ALIASES) if include_rul else [])
    mapping = {k: None for k in targets}
    used = set()
    for norm, orig in norm_user.items():
        key = lookup.get(norm)
        if key in mapping and mapping[key] is None and orig not in used:
            mapping[key] = orig
            used.add(orig)
    return mapping


if __name__ == "__main__":
    print(f"{len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical features; "
          f"{len(TARGETS)} targets; risk bins: {[b[0] for b in RISK_BINS]}")
