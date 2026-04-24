import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =========================
# 1 LOAD TRAIN DATA
# =========================
train_df = pd.read_csv("train.csv")

# Numeric columns
num_cols = ['age','stress_level','anxiety_level','depression_score',
            'maslow_physiological','maslow_safety','maslow_social',
            'maslow_esteem','maslow_self_actualization','sleep_hours',
            'exercise_frequency','caffeine_intake','screen_time_hours',
            'duration_days','improvement_score']

train_df[num_cols] = train_df[num_cols].apply(pd.to_numeric, errors='coerce')

# =========================
# 2 FEATURE ENGINEERING
# =========================
train_df['psych_indicators'] = train_df['stress_level'] + train_df['anxiety_level'] + train_df['depression_score']
train_df['maslow_satisfaction_index'] = (
    train_df['maslow_physiological'] + train_df['maslow_safety'] +
    train_df['maslow_social'] + train_df['maslow_esteem'] +
    train_df['maslow_self_actualization']
)

# =========================
# 3 ENCODING CATEGORICALS
# =========================
train_df["gender"] = train_df["gender"].map({"Male":0, "Female":1, "Other":2})
train_df["diet_type"] = train_df["diet_type"].map({"Balanced":0, "Light":1, "Spicy":2})
train_df["side_effects"] = train_df["side_effects"].map({True:1, False:0})

# =========================
# 4 PRAKRITI & VIKRITI
# =========================
for col in ["prakriti", "vikriti"]:
    for type_name in ["Vata","Pitta","Kapha"]:
        train_df[f"{col.lower()}_{type_name.lower()}"] = 0
        train_df.loc[train_df[col].str.contains(type_name, na=False), f"{col.lower()}_{type_name.lower()}"] = 1

# =========================
# 5 MULTI-LABEL COLUMNS
# =========================
def multi_label_dummies(df, column, prefix):
    df[column] = df[column].fillna("")
    dummies = (df[column].str.strip("[]")
               .str.replace("'", "")
               .str.get_dummies(sep=", ")
               .add_prefix(prefix))
    df = pd.concat([df, dummies], axis=1)
    df.drop(column, axis=1, inplace=True)
    return df

train_df = multi_label_dummies(train_df, "herbs", "herb_")
train_df = multi_label_dummies(train_df, "diet_advice", "diet_")
train_df = multi_label_dummies(train_df, "lifestyle_advice", "life_")

# =========================
# 6 DROP UNUSED COLUMNS
# =========================
train_df.drop(["patient_id","prakriti","vikriti"], axis=1, inplace=True)

# =========================
# 7 SPLIT FEATURES & TARGET
# =========================
y = train_df["improvement_score"]
X = train_df.drop("improvement_score", axis=1)

# =========================
# 8 TRAIN MODEL
# =========================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42, n_estimators=200)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print("Validation RMSE:", rmse)

# =========================
# 9 LOAD TEST DATA
# =========================
test_df = pd.read_csv("test.csv")
test_ids = test_df["patient_id"]

# Numeric columns
test_df[num_cols[:-1]] = test_df[num_cols[:-1]].apply(pd.to_numeric, errors='coerce')

# Feature engineering
test_df['psych_indicators'] = test_df['stress_level'] + test_df['anxiety_level'] + test_df['depression_score']
test_df['maslow_satisfaction_index'] = (
    test_df['maslow_physiological'] + test_df['maslow_safety'] +
    test_df['maslow_social'] + test_df['maslow_esteem'] +
    test_df['maslow_self_actualization']
)

# Encoding
test_df["gender"] = test_df["gender"].map({"Male":0, "Female":1, "Other":2})
test_df["diet_type"] = test_df["diet_type"].map({"Balanced":0, "Light":1, "Spicy":2})
test_df["side_effects"] = test_df["side_effects"].map({True:1, False:0})

# PRAKRITI & VIKRITI
for col in ["prakriti", "vikriti"]:
    for type_name in ["Vata","Pitta","Kapha"]:
        test_df[f"{col.lower()}_{type_name.lower()}"] = 0
        test_df.loc[test_df[col].str.contains(type_name, na=False), f"{col.lower()}_{type_name.lower()}"] = 1

# MULTI-LABEL
test_df = multi_label_dummies(test_df, "herbs", "herb_")
test_df = multi_label_dummies(test_df, "diet_advice", "diet_")
test_df = multi_label_dummies(test_df, "lifestyle_advice", "life_")

# Drop unused
test_df.drop(["patient_id","prakriti","vikriti"], axis=1, inplace=True)

# Ensure same columns as training
test_df = test_df.reindex(columns=X.columns, fill_value=0)

# =========================
#  PREDICT
# =========================
test_predictions = model.predict(test_df)

# =========================
#  CREATE SUBMISSION
# =========================
submission = pd.DataFrame({
    "patient_id": test_ids,
    "improvement_score": test_predictions
})

submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")