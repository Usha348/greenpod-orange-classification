
import pandas as pd

results = {
    "Model": ["Random Forest", "CNN"],
    "Approach": [
        "Traditional Machine Learning",
        "Deep Learning"
    ],
    "Test Accuracy (%)": [91.92, 93.94]
}
df = pd.DataFrame(results)

print("\nModel Comparison")
print("================")
print(df.to_string(index=False))
