
import pandas as pd
from pathlib import Path

file_path = Path(
    "ai4i+2020+predictive+maintenance+dataset"
) / "ai4i2020.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMachine failure counts:")
print(df["Machine failure"].value_counts())

print("\nMissing values per column:")
print(df.isnull().sum())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the dataset
file_path = (
    Path("ai4i+2020+predictive+maintenance+dataset")
    / "ai4i2020.csv"
)

df = pd.read_csv(file_path)

sns.set_theme(style="whitegrid")

# Create four graphs
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Graph 1: Machine failure counts
failure_counts = df["Machine failure"].value_counts().sort_index()

axes[0, 0].bar(
    ["No Failure (0)", "Failure (1)"],
    [failure_counts.get(0, 0), failure_counts.get(1, 0)]
)
axes[0, 0].set_title("Machine Failure Counts")
axes[0, 0].set_ylabel("Number of Records")

# Graph 2: Air temperature distribution
sns.histplot(
    data=df,
    x="Air temperature [K]",
    bins=30,
    kde=True,
    ax=axes[0, 1]
)
axes[0, 1].set_title("Air Temperature Distribution")

# Graph 3: Rotational speed vs torque
sns.scatterplot(
    data=df,
    x="Rotational speed [rpm]",
    y="Torque [Nm]",
    hue="Machine failure",
    palette={0: "steelblue", 1: "red"},
    alpha=0.6,
    s=25,
    ax=axes[1, 0]
)
axes[1, 0].set_title("Rotational Speed vs Torque")
axes[1, 0].legend(title="Failure (0/1)")

# Graph 4: Correlation heatmap
numeric_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
]

correlation = df[numeric_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    ax=axes[1, 1]
)
axes[1, 1].set_title("Sensor Feature Correlations")

plt.tight_layout()

# Save the graphs in your project folder
output_file = "ai4i_dataset_plots.png"
plt.savefig(output_file, dpi=200, bbox_inches="tight")

print("Graphs saved successfully!")
print("File:", Path(output_file).resolve())

# Display the graphs
plt.show()
