import pandas as pd
import sqlite3

# 1. CSV read karke SQLite Database (.db) me load karein
df = pd.read_csv('sales_data.csv')
conn = sqlite3.connect('analytics.db')
df.to_sql('passengers', conn, if_exists='replace', index=False)

print("=== [SQL Query 1: Top 5 Highest Fare Payers] ===")
q1 = """
SELECT Name, Pclass, Fare, Survived 
FROM passengers 
ORDER BY Fare DESC 
LIMIT 5;
"""
print(pd.read_sql_query(q1, conn))

print("\n=== [SQL Query 2: Class-wise Passenger Count & Avg Fare] ===")
q2 = """
SELECT 
    Pclass,
    COUNT(*) AS Total_Passengers,
    ROUND(AVG(Fare), 2) AS Avg_Fare,
    ROUND(AVG(Survived) * 100, 2) AS Survival_Percentage
FROM passengers
GROUP BY Pclass;
"""
print(pd.read_sql_query(q2, conn))

conn.close()
