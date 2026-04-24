import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# First lets have the data over here as a dataframe
train_df = pd.read_csv("train.csv")
# print(train_df)

# Now lets clean the data a bit, or if clean lets check few things out
# Is there a repeated patient_id?
# print(train_df["patient_id"].is_unique)     # We got True(so no repeating ids)

'''
# Is there any Null values?
print(train_df.isnull())                    # This would give T/F in entire df but we have too many values so
print(train_df.isnull().sum())              # So we got all filled values in here
'''

# Now as we have numbers in our data we must check if we have NaN values in those columns or not
cols = ['age','stress_level','anxiety_level','depression_score',
        'maslow_physiological','maslow_safety','maslow_social',
        'maslow_esteem','maslow_self_actualization','sleep_hours',
        'exercise_frequency','caffeine_intake','screen_time_hours',
        'duration_days','improvement_score']

train_df[cols] = train_df[cols].apply(pd.to_numeric, errors='coerce')      #Coerce is used when a data isn't converted then it must become NaN



# Lets now understand target variable
'''
# Minimum value of that
print(train_df['improvement_score'].max())
print(train_df['improvement_score'].min())
print(train_df["improvement_score"].mean())
print(train_df["improvement_score"].median())
print(train_df["improvement_score"].std())
# print(train_df["improvement_score"].value_counts())   # distribution
'''
# Lets check it visually as well
# train_df['improvement_score'].hist()
# plt.show()


# Now we will start feature engineering, checking trends and correlation to that of the target variable
# We'll now make a correlation matrix checking for each numerical variable with improvement score

# print(train_df[['age','stress_level','anxiety_level','depression_score',
#         'sleep_hours','exercise_frequency','caffeine_intake','screen_time_hours',
#         'duration_days', 'maslow_physiological','maslow_safety','maslow_social','maslow_esteem','maslow_self_actualization','improvement_score']].corr()['improvement_score'])



train_df['pysch_indicators'] = train_df['stress_level'] + train_df['anxiety_level'] + train_df['depression_score']
train_df['maslow_satisfaction_index'] = train_df['maslow_physiological'] + train_df['maslow_safety'] + train_df['maslow_social'] + train_df['maslow_esteem'] + train_df['maslow_self_actualization']

# print(train_df)

#print(train_df[['pysch_indicators', 'maslow_satisfaction_index', 'improvement_score']].corr()['improvement_score'])

# Encode gender
train_df["gender"] = train_df["gender"].map({
    "Male": 0,
    "Female": 1,
    "Other": 2
})

# Encode diet type
train_df["diet_type"] = train_df["diet_type"].map({
    "Balanced": 0,
    "Light": 1,
    "Spicy": 2
})

# Encode side effects
train_df["side_effects"] = train_df["side_effects"].map({
    True: 1,
    False: 0
})

# Create empty columns
train_df["prakriti_vata"] = 0
train_df["prakriti_pitta"] = 0
train_df["prakriti_kapha"] = 0


# Fill values
train_df.loc[train_df["prakriti"].str.contains("Vata"), "prakriti_vata"] = 1
train_df.loc[train_df["prakriti"].str.contains("Pitta"), "prakriti_pitta"] = 1
train_df.loc[train_df["prakriti"].str.contains("Kapha"), "prakriti_kapha"] = 1

train_df["vikriti_vata"] = 0
train_df["vikriti_pitta"] = 0
train_df["vikriti_kapha"] = 0

train_df.loc[train_df["vikriti"].str.contains("Vata"), "vikriti_vata"] = 1
train_df.loc[train_df["vikriti"].str.contains("Pitta"), "vikriti_pitta"] = 1
train_df.loc[train_df["vikriti"].str.contains("Kapha"), "vikriti_kapha"] = 1

# You know why we did this making many new columns instead of marking 1,2 or 3 to them?
# That is coz then the model would think according to superiorarity as 1 < 2 < 3

# Removing [] and having the data in commas
train_df["herbs"] = train_df["herbs"].fillna("")

herb_dummies = (
    train_df["herbs"]
    .str.strip("[]")
    .str.replace("'", "")
    .str.get_dummies(sep=", ")
    .add_prefix("herb_")
)

train_df = pd.concat([train_df, herb_dummies], axis=1)

train_df.drop("herbs", axis=1, inplace=True)

# Drop unnecessary columns
train_df.drop(["patient_id","prakriti","vikriti"], axis=1, inplace=True)

# print(train_df.shape)
# print(train_df.columns)

train_df["diet_advice"] = train_df["diet_advice"].fillna("")

diet_dummies = (
    train_df["diet_advice"]
    .str.strip("[]")
    .str.replace("'", "")
    .str.get_dummies(sep=", ")
    .add_prefix("diet_")
)

train_df = pd.concat([train_df, diet_dummies], axis=1)

train_df.drop("diet_advice", axis=1, inplace=True)



train_df["lifestyle_advice"] = train_df["lifestyle_advice"].fillna("")

lifestyle_dummies = (
    train_df["lifestyle_advice"]
    .str.strip("[]")
    .str.replace("'", "")
    .str.get_dummies(sep=", ")
    .add_prefix("life_")
)

train_df = pd.concat([train_df, lifestyle_dummies], axis=1)

train_df.drop("lifestyle_advice", axis=1, inplace=True)




y = train_df["improvement_score"]
X = train_df.drop("improvement_score", axis=1)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print("RMSE:", rmse)


importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False).head(10))