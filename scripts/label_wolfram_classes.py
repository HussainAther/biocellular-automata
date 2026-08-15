# scripts/label_wolfram_classes.py

import pandas as pd

def classify_row(entropy, activity):
    """
    Heuristic thresholds based on empirical observation:
    - Class I: low entropy, low activity → uniform / frozen
    - Class II: low entropy, medium activity → periodic
    - Class III: high entropy, high activity → chaotic
    - Class IV: medium entropy, medium activity → complex
    """
    if entropy < 0.2 and activity < 0.2:
        return 1  # Class I
    elif entropy < 0.5 and activity < 0.6:
        return 2  # Class II
    elif entropy > 0.8 and activity > 0.8:
        return 3  # Class III
    else:
        return 4  # Class IV

def main():
    df = pd.read_csv("data/elementary_rules_dataset.csv")
    print("[*] Labeling dataset using entropy/activity thresholds...")

    df["wolfram_class"] = df.apply(
        lambda row: classify_row(row["entropy"], row["activity"]),
        axis=1
    )

    output_path = "data/elementary_rules_labeled.csv"
    df.to_csv(output_path, index=False)
    print(f"[✓] Labeled dataset saved to {output_path}")

if __name__ == "__main__":
    main()

