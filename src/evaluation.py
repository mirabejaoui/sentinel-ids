"""
Protocole d'évaluation : découpage des données et calcul des métriques.

Deux découpages :
- temporal_split : découpage principal. Pour chaque type de flux (chaque
  étiquette de chaque jour), les premiers 70 % vont en entraînement, les
  15 % suivants en validation et les 15 % derniers en test.
- random_split : découpage aléatoire classique, utilisé seulement pour
  montrer à quel point il surestime les performances.

Le scénario «attaques inconnues» (leave-one-attack-out) est traité en phase 5.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

META = ["Label", "day", "day_order", "row_in_file", "family", "is_attack"]


def temporal_split(df: pd.DataFrame, train=0.70, val=0.15) -> pd.Series:
    """Renvoie une colonne 'train' / 'val' / 'test' alignée sur df."""
    # Position de chaque flux dans son groupe (même jour, même étiquette),
    # entre 0 et 1, dans l'ordre du fichier (approximativement chronologique)
    position = df.groupby(["day", "Label"], observed=True)["row_in_file"].rank(
        method="first", pct=True
    )
    split = np.select([position <= train, position <= train + val], ["train", "val"], "test")
    return pd.Series(split, index=df.index, name="split")


def random_split(df: pd.DataFrame, train=0.70, val=0.15, seed=42) -> pd.Series:
    """Découpage aléatoire stratifié par étiquette (pour comparaison uniquement)."""
    idx_train, idx_rest = train_test_split(
        df.index, train_size=train, stratify=df["Label"], random_state=seed
    )
    rest = df.loc[idx_rest, "Label"]
    idx_val, _ = train_test_split(
        idx_rest, train_size=val / (1 - train), stratify=rest, random_state=seed
    )
    split = pd.Series("test", index=df.index, name="split")
    split[idx_train] = "train"
    split[idx_val] = "val"
    return split


def false_positive_rate(y_true_binary, y_pred_binary) -> float:
    """Part du trafic normal signalé à tort comme attaque."""
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    return float(fp / (fp + tn))


def evaluate_binary(y_true, y_pred, y_score=None) -> dict:
    """Métriques pour la tâche normal (0) / attaque (1)."""
    results = {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "recall_attack": float(((y_pred == 1) & (y_true == 1)).sum() / (y_true == 1).sum()),
        "false_positive_rate": false_positive_rate(y_true, y_pred),
    }
    if y_score is not None:
        results["pr_auc"] = average_precision_score(y_true, y_score)
    return results


def evaluate_multiclass(y_true, y_pred) -> pd.DataFrame:
    """Précision, rappel et F1 par famille, plus les moyennes."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.round(4)