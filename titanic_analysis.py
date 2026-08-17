import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Dataset Load
df = pd.read_csv('sales_data.csv')

# 2. Basic Info & Metrics
print("=== Basic Dataset Info ===")
print(df.info())

print("\n=== Overall Survival Rate ===")
survival_rate = df['Survived'].mean() * 100
print(f"Total Survival Rate: {survival_rate:.2f}%")

print("\n=== Survival Rate by Gender ===")
gender_survival = df.groupby('Sex')['Survived'].value_counts(normalize=True).unstack() * 100
print(gender_survival)

# 3. Data Cleaning
df['Age_Cleaned'] = df['Age'].fillna(df['Age'].median())

# 4. Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(ax=axes[0], x='Pclass', y='Survived', hue='Sex', data=df, palette='Set2')
axes[0].set_title('Survival Rate by Passenger Class & Gender')
axes[0].set_ylabel('Survival Rate')

sns.histplot(ax=axes[1], data=df, x='Age_Cleaned', hue='Survived', kde=True, bins=25, palette='coolwarm')
axes[1].set_title('Age Distribution by Survival')

plt.tight_layout()
plt.savefig('titanic_chart.png')
print("\n[SUCCESS] Analysis complete! Chart saved as 'titanic_chart.png'")
plt.show()
