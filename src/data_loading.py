"""
Chargement et nettoyage de CICIDS2017 (version MachineLearningCSV).

Usage, depuis la racine du projet :
    python -m src.data_loading

Entrée  : data/raw/*.csv (les 8 fichiers originaux)
Sortie  : data/processed/cicids2017_clean.parquet

Le script lit les fichiers un par un et passe les colonnes en float32
pour limiter la mémoire.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "processed" / "cicids2017_clean.parquet"

# Jour et demi-journée de chaque fichier : sert au découpage temporel (phase 3),
# car cette version du dataset ne contient pas de colonne Timestamp.
DAY_ORDER = {
    "Monday-WorkingHours": (0, "Monday"),
    "Tuesday-WorkingHours": (1, "Tuesday"),
    "Wednesday-workingHours": (2, "Wednesday"),
    "Thursday-WorkingHours-Morning-WebAttacks": (3, "Thursday-AM"),
    "Thursday-WorkingHours-Afternoon-Infilteration": (4, "Thursday-PM"),
    "Friday-WorkingHours-Morning": (5, "Friday-AM"),
    "Friday-WorkingHours-Afternoon-PortScan": (6, "Friday-PM-PortScan"),
    "Friday-WorkingHours-Afternoon-DDos": (7, "Friday-PM-DDoS"),
}

# Étiquette fine -> famille d'attaques.
# Heartbleed (11 flux) est rangé avec DoS/DDoS, comme dans une bonne partie de
# la littérature : c'est discutable (c'est un exploit, pas un déni de service),
# à mentionner dans le rapport.
FAMILY = {
    "BENIGN": "Normal",
    "DoS Hulk": "DoS/DDoS",
    "DoS GoldenEye": "DoS/DDoS",
    "DoS slowloris": "DoS/DDoS",
    "DoS Slowhttptest": "DoS/DDoS",
    "DDoS": "DoS/DDoS",
    "Heartbleed": "DoS/DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Bot": "Botnet",
    "Infiltration": "Infiltration",
}


def clean_label(label: str) -> str:
    """Corrige le caractère mal encodé des attaques web ('Web Attack � XSS')."""
    label = label.strip()
    return re.sub(r"^Web Attack\s*\S\s*", "Web Attack - ", label)


def load_file(path: Path) -> tuple[pd.DataFrame, int, int]:
    # On lit directement en float32 : la lecture par défaut en float64 doublerait la mémoire
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: np.float32 for c in header if c.strip() != "Label"}
    df = pd.read_csv(path, dtype=dtypes, encoding="utf-8", encoding_errors="replace")

    # 1. Noms de colonnes : espaces parasites + colonne dupliquée
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["Fwd Header Length.1"], errors="ignore")

    # 2. Étiquettes
    df["Label"] = df["Label"].astype(str).map(clean_label)

    # 3. Le port est un identifiant, pas une grandeur : entier
    df["Destination Port"] = df["Destination Port"].astype(np.int32)

    # 4. Métadonnées pour le protocole d'évaluation
    order, day = DAY_ORDER[path.stem.replace(".pcap_ISCX", "")]
    df["day"] = day
    df["day_order"] = np.int8(order)
    df["row_in_file"] = np.arange(len(df), dtype=np.int32)  # ordre approx. chronologique

    # 5. Inf/NaN et doublons internes au fichier, traités ici pour limiter la mémoire
    num = df.select_dtypes("number").columns
    bad = ~np.isfinite(df[num].to_numpy(dtype=np.float32)).all(axis=1)
    df = df[~bad]
    # Empreinte de chaque ligne (hors métadonnées) : bien plus léger en mémoire
    # que df.duplicated() sur 2,8 millions de lignes et 70 colonnes
    row_hash = pd.util.hash_pandas_object(
        df.drop(columns=["day", "day_order", "row_in_file"]), index=False
    )
    dup = row_hash.duplicated().to_numpy()
    df = df[~dup].copy()
    df["_hash"] = row_hash[~dup].to_numpy()
    return df, int(bad.sum()), int(dup.sum())


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv"))
    if len(files) != 8:
        raise SystemExit(f"{len(files)} fichiers trouvés dans {RAW_DIR}, 8 attendus.")

    parts, n0, n_bad, n_dup = [], 0, 0, 0
    for f in files:
        part, bad, dup = load_file(f)
        n0 += len(part) + bad + dup
        n_bad += bad
        n_dup += dup
        print(f"  {f.name:<62} {len(part):>9,} lignes gardées")
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f"\nTotal brut : {n0:,} lignes")
    print(f"Lignes avec Inf/NaN supprimées : {n_bad:,}")

    # 6. Doublons entre fichiers (même flux présent deux jours différents)
    meta = ["day", "day_order", "row_in_file"]
    dup = df["_hash"].duplicated().to_numpy()
    n_dup += int(dup.sum())
    df = df[~dup].drop(columns="_hash")
    print(f"Doublons supprimés            : {n_dup:,}")

    # 7. Variables constantes
    features = df.columns.drop(["Label"] + meta)
    constant = [c for c in features if df[c].nunique() <= 1]
    df = df.drop(columns=constant)
    print(f"Colonnes constantes supprimées : {len(constant)} -> {constant}")

    # 8. Cibles : binaire et familles
    unknown = set(df["Label"]) - set(FAMILY)
    if unknown:
        raise SystemExit(f"Étiquettes non prévues : {unknown}")
    df["family"] = df["Label"].map(FAMILY).astype("category")
    df["is_attack"] = (df["family"] != "Normal").astype(np.int8)
    df["Label"] = df["Label"].astype("category")
    df["day"] = df["day"].astype("category")

    df = df.reset_index(drop=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)

    n_feat = df.shape[1] - len(meta) - 3
    print(f"\nFinal : {len(df):,} lignes ({len(df)/n0:.1%} du brut), {n_feat} variables")
    print(f"Mémoire : {df.memory_usage(deep=True).sum()/1e9:.2f} Go -> {OUT_FILE}")
    print("\nRépartition par famille :")
    print(df["family"].value_counts().to_string())


if __name__ == "__main__":
    main()