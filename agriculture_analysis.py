import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("seasonal_agriculture_performance_dataset.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Information:")
df.info()

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 2. DATA TYPES
# ============================================================

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

df["Rainfall_mm"] = df["Rainfall_mm"].fillna(
    df["Rainfall_mm"].median()
)

df["Soil_Moisture_pct"] = df["Soil_Moisture_pct"].fillna(
    df["Soil_Moisture_pct"].median()
)

df["Yield_Tonnes_Ha"] = df["Yield_Tonnes_Ha"].fillna(
    df["Yield_Tonnes_Ha"].median()
)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ============================================================
# 4. STATISTICAL SUMMARY
# ============================================================

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 5. UNIQUE VALUES
# ============================================================

print("\nUnique States:")
print(df["State"].unique())

print("\nUnique Crops:")
print(df["Crop"].unique())

print("\nUnique Seasons:")
print(df["Season"].unique())

print("\nIrrigation Methods:")
print(df["Irrigation_Method"].unique())


# ============================================================
# 6. CROP-WISE PERFORMANCE ANALYSIS
# ============================================================

crop_analysis = df.groupby("Crop").agg({
    "Yield_Tonnes_Ha": "mean",
    "Production_Tonnes": "sum",
    "Revenue_INR": "mean",
    "Profit_INR": "mean"
}).round(2)

print("\nCrop-wise Performance:")
print(crop_analysis)

best_profit_crop = crop_analysis["Profit_INR"].idxmax()

print("\nMost Profitable Crop:", best_profit_crop)
print(
    "Average Profit:",
    crop_analysis.loc[best_profit_crop, "Profit_INR"]
)

best_yield_crop = crop_analysis["Yield_Tonnes_Ha"].idxmax()

print("\nHighest Yield Crop:", best_yield_crop)
print(
    "Average Yield:",
    crop_analysis.loc[best_yield_crop, "Yield_Tonnes_Ha"]
)


# ============================================================
# 7. STATE-WISE PERFORMANCE ANALYSIS
# ============================================================

state_analysis = df.groupby("State").agg({
    "Yield_Tonnes_Ha": "mean",
    "Production_Tonnes": "sum",
    "Revenue_INR": "mean",
    "Profit_INR": "mean"
}).round(2)

print("\nState-wise Performance:")
print(state_analysis)

best_state_profit = state_analysis["Profit_INR"].idxmax()

print("\nMost Profitable State:", best_state_profit)
print(
    "Average Profit:",
    state_analysis.loc[best_state_profit, "Profit_INR"]
)

best_state_production = state_analysis["Production_Tonnes"].idxmax()

print("\nHighest Production State:", best_state_production)
print(
    "Total Production:",
    state_analysis.loc[
        best_state_production,
        "Production_Tonnes"
    ]
)


# ============================================================
# 8. SEASON-WISE PERFORMANCE ANALYSIS
# MAIN FOCUS OF PROJECT
# ============================================================

season_analysis = df.groupby("Season").agg({
    "Yield_Tonnes_Ha": "mean",
    "Production_Tonnes": "sum",
    "Revenue_INR": "mean",
    "Profit_INR": "mean"
}).round(2)

print("\nSeason-wise Performance:")
print(season_analysis)

best_season_profit = season_analysis["Profit_INR"].idxmax()

print("\nMost Profitable Season:", best_season_profit)
print(
    "Average Profit:",
    season_analysis.loc[best_season_profit, "Profit_INR"]
)

best_season_yield = season_analysis["Yield_Tonnes_Ha"].idxmax()

print("\nHighest Yield Season:", best_season_yield)
print(
    "Average Yield:",
    season_analysis.loc[best_season_yield, "Yield_Tonnes_Ha"]
)

best_season_production = season_analysis["Production_Tonnes"].idxmax()

print("\nHighest Production Season:", best_season_production)
print(
    "Total Production:",
    season_analysis.loc[
        best_season_production,
        "Production_Tonnes"
    ]
)


# ============================================================
# 9. IRRIGATION METHOD ANALYSIS
# ============================================================

irrigation_analysis = df.groupby("Irrigation_Method").agg({
    "Yield_Tonnes_Ha": "mean",
    "Production_Tonnes": "sum",
    "Profit_INR": "mean",
    "Water_Efficiency_t_per_1000m3": "mean"
}).round(2)

print("\nIrrigation Method Performance:")
print(irrigation_analysis)

best_irrigation_yield = irrigation_analysis[
    "Yield_Tonnes_Ha"
].idxmax()

print(
    "\nHighest Yield Irrigation Method:",
    best_irrigation_yield
)

print(
    "Average Yield:",
    irrigation_analysis.loc[
        best_irrigation_yield,
        "Yield_Tonnes_Ha"
    ]
)

best_irrigation_water = irrigation_analysis[
    "Water_Efficiency_t_per_1000m3"
].idxmax()

print(
    "\nMost Water-Efficient Irrigation Method:",
    best_irrigation_water
)

print(
    "Water Efficiency:",
    irrigation_analysis.loc[
        best_irrigation_water,
        "Water_Efficiency_t_per_1000m3"
    ]
)


# ============================================================
# 10. FERTILIZER ANALYSIS
# ============================================================

print("\nFertilizer Analysis:")

print(
    "Average Fertilizer Used (kg/ha):",
    round(df["Fertilizer_kg_ha"].mean(), 2)
)

print(
    "Average Yield (tonnes/ha):",
    round(df["Yield_Tonnes_Ha"].mean(), 2)
)

print(
    "Average Profit (INR):",
    round(df["Profit_INR"].mean(), 2)
)

fertilizer_bins = pd.cut(
    df["Fertilizer_kg_ha"],
    bins=4
)

fertilizer_analysis = df.groupby(
    fertilizer_bins,
    observed=True
).agg({
    "Yield_Tonnes_Ha": "mean",
    "Profit_INR": "mean"
}).round(2)

print("\nFertilizer Usage Group Performance:")
print(fertilizer_analysis)


# ============================================================
# 11. PESTICIDE & DISEASE/PEST RISK ANALYSIS
# ============================================================

print("\nPesticide & Disease Risk Analysis:")

print(
    "Average Pesticide Used (L/ha):",
    round(df["Pesticide_Litre_ha"].mean(), 2)
)

print(
    "Average Disease/Pest Risk (%):",
    round(df["Disease_Pest_Risk_pct"].mean(), 2)
)

risk_analysis = df.groupby(
    pd.cut(df["Pesticide_Litre_ha"], bins=4)
).agg({
    "Disease_Pest_Risk_pct": "mean",
    "Yield_Tonnes_Ha": "mean",
    "Profit_INR": "mean"
}).round(2)

print("\nPesticide Usage Group Performance:")
print(risk_analysis)

highest_risk_group = risk_analysis[
    "Disease_Pest_Risk_pct"
].idxmax()

print(
    "\nHighest Disease/Pest Risk Group:",
    highest_risk_group
)

print(
    "Average Risk (%):",
    risk_analysis.loc[
        highest_risk_group,
        "Disease_Pest_Risk_pct"
    ]
)


# ============================================================
# 12. WEATHER FACTORS ANALYSIS
# ============================================================

print("\nWeather Factors Analysis:")

weather_analysis = df[[
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Yield_Tonnes_Ha"
]].corr().round(2)

print("\nCorrelation between Weather Factors and Yield:")
print(weather_analysis)

print(
    "\nRainfall-Yield Correlation:",
    weather_analysis.loc[
        "Rainfall_mm",
        "Yield_Tonnes_Ha"
    ]
)

print(
    "Temperature-Yield Correlation:",
    weather_analysis.loc[
        "Avg_Temperature_C",
        "Yield_Tonnes_Ha"
    ]
)

print(
    "Humidity-Yield Correlation:",
    weather_analysis.loc[
        "Humidity_pct",
        "Yield_Tonnes_Ha"
    ]
)


# ============================================================
# 13. SOIL & NUTRIENT ANALYSIS
# ============================================================

print("\nSoil & Nutrient Analysis:")

soil_analysis = df[[
    "Soil_pH",
    "Soil_Moisture_pct",
    "Nitrogen_kg_ha",
    "Phosphorus_kg_ha",
    "Potassium_kg_ha",
    "Yield_Tonnes_Ha"
]].corr().round(2)

print("\nCorrelation between Soil/Nutrients and Yield:")
print(soil_analysis)

print(
    "\nSoil pH-Yield Correlation:",
    soil_analysis.loc[
        "Soil_pH",
        "Yield_Tonnes_Ha"
    ]
)

print(
    "Soil Moisture-Yield Correlation:",
    soil_analysis.loc[
        "Soil_Moisture_pct",
        "Yield_Tonnes_Ha"
    ]
)

print(
    "Nitrogen-Yield Correlation:",
    soil_analysis.loc[
        "Nitrogen_kg_ha",
        "Yield_Tonnes_Ha"
    ]
)

print(
    "Phosphorus-Yield Correlation:",
    soil_analysis.loc[
        "Phosphorus_kg_ha",
        "Yield_Tonnes_Ha"
    ]
)

print(
    "Potassium-Yield Correlation:",
    soil_analysis.loc[
        "Potassium_kg_ha",
        "Yield_Tonnes_Ha"
    ]
)


# ============================================================
# 14. SEED QUALITY ANALYSIS
# ============================================================

print("\nSeed Quality Analysis:")

print(
    "Average Seed Quality Score:",
    round(df["Seed_Quality_Score"].mean(), 2)
)

print(
    "Average Yield (tonnes/ha):",
    round(df["Yield_Tonnes_Ha"].mean(), 2)
)

print(
    "Average Profit (INR):",
    round(df["Profit_INR"].mean(), 2)
)

seed_analysis = df[[
    "Seed_Quality_Score",
    "Yield_Tonnes_Ha",
    "Profit_INR"
]].corr().round(2)

print("\nCorrelation between Seed Quality and Performance:")
print(seed_analysis)

print(
    "\nSeed Quality-Yield Correlation:",
    seed_analysis.loc[
        "Seed_Quality_Score",
        "Yield_Tonnes_Ha"
    ]
)

print(
    "Seed Quality-Profit Correlation:",
    seed_analysis.loc[
        "Seed_Quality_Score",
        "Profit_INR"
    ]
)


# ============================================================
# 15. FARM AREA ANALYSIS
# ============================================================

print("\nFarm Area Analysis:")

print(
    "Average Farm Area (hectares):",
    round(df["Farm_Area_Hectares"].mean(), 2)
)

print(
    "Average Production (tonnes):",
    round(df["Production_Tonnes"].mean(), 2)
)

print(
    "Average Revenue (INR):",
    round(df["Revenue_INR"].mean(), 2)
)

print(
    "Average Profit (INR):",
    round(df["Profit_INR"].mean(), 2)
)

farm_area_analysis = df[[
    "Farm_Area_Hectares",
    "Production_Tonnes",
    "Revenue_INR",
    "Profit_INR"
]].corr().round(2)

print("\nCorrelation between Farm Area and Performance:")
print(farm_area_analysis)

print(
    "\nFarm Area-Production Correlation:",
    farm_area_analysis.loc[
        "Farm_Area_Hectares",
        "Production_Tonnes"
    ]
)

print(
    "Farm Area-Revenue Correlation:",
    farm_area_analysis.loc[
        "Farm_Area_Hectares",
        "Revenue_INR"
    ]
)

print(
    "Farm Area-Profit Correlation:",
    farm_area_analysis.loc[
        "Farm_Area_Hectares",
        "Profit_INR"
    ]
)


# ============================================================
# 16. MARKET PRICE ANALYSIS
# ============================================================

print("\nMarket Price Analysis:")

print(
    "Average Market Price (INR/tonne):",
    round(df["Market_Price_INR_Tonne"].mean(), 2)
)

print(
    "Minimum Market Price (INR/tonne):",
    df["Market_Price_INR_Tonne"].min()
)

print(
    "Maximum Market Price (INR/tonne):",
    df["Market_Price_INR_Tonne"].max()
)

market_analysis = df[[
    "Market_Price_INR_Tonne",
    "Revenue_INR",
    "Profit_INR"
]].corr().round(2)

print("\nCorrelation between Market Price and Performance:")
print(market_analysis)

print(
    "\nMarket Price-Revenue Correlation:",
    market_analysis.loc[
        "Market_Price_INR_Tonne",
        "Revenue_INR"
    ]
)

print(
    "Market Price-Profit Correlation:",
    market_analysis.loc[
        "Market_Price_INR_Tonne",
        "Profit_INR"
    ]
)


# ============================================================
# 17. WATER USAGE & EFFICIENCY ANALYSIS
# ============================================================

print("\nWater Usage & Efficiency Analysis:")

print(
    "Average Water Used (m3):",
    round(df["Water_Used_m3"].mean(), 2)
)

print(
    "Average Water Efficiency (tonnes/1000m3):",
    round(
        df["Water_Efficiency_t_per_1000m3"].mean(),
        2
    )
)

water_analysis = df.groupby("Irrigation_Method").agg({
    "Water_Used_m3": "mean",
    "Water_Efficiency_t_per_1000m3": "mean",
    "Yield_Tonnes_Ha": "mean"
}).round(2)

print("\nWater Performance by Irrigation Method:")
print(water_analysis)

best_water_method = water_analysis[
    "Water_Efficiency_t_per_1000m3"
].idxmax()

print(
    "\nMost Water-Efficient Irrigation Method:",
    best_water_method
)

print(
    "Water Efficiency:",
    water_analysis.loc[
        best_water_method,
        "Water_Efficiency_t_per_1000m3"
    ]
)


# ============================================================
# 18. SEASON-WISE AVERAGE YIELD VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    season_analysis.index,
    season_analysis["Yield_Tonnes_Ha"]
)

plt.title("Average Yield by Season")
plt.xlabel("Season")
plt.ylabel("Average Yield (Tonnes/Ha)")

plt.tight_layout()
plt.show()


# ============================================================
# 19. SEASON-WISE AVERAGE PROFIT VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    season_analysis.index,
    season_analysis["Profit_INR"]
)

plt.title("Average Profit by Season")
plt.xlabel("Season")
plt.ylabel("Average Profit (INR)")

plt.tight_layout()
plt.show()


# ============================================================
# 20. SEASON-WISE TOTAL PRODUCTION VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    season_analysis.index,
    season_analysis["Production_Tonnes"]
)

plt.title("Total Production by Season")
plt.xlabel("Season")
plt.ylabel("Total Production (Tonnes)")

plt.tight_layout()
plt.show()


# ============================================================
# 21. IRRIGATION METHOD VS AVERAGE YIELD
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    irrigation_analysis.index,
    irrigation_analysis["Yield_Tonnes_Ha"]
)

plt.title("Average Yield by Irrigation Method")
plt.xlabel("Irrigation Method")
plt.ylabel("Average Yield (Tonnes/Ha)")

plt.tight_layout()
plt.show()


# ============================================================
# 22. WATER EFFICIENCY BY IRRIGATION METHOD
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    water_analysis.index,
    water_analysis["Water_Efficiency_t_per_1000m3"]
)

plt.title("Water Efficiency by Irrigation Method")
plt.xlabel("Irrigation Method")
plt.ylabel("Water Efficiency (Tonnes/1000 m³)")

plt.tight_layout()
plt.show()


# ============================================================
# 23. CROP-WISE AVERAGE PROFIT VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    crop_analysis.index,
    crop_analysis["Profit_INR"]
)

plt.title("Average Profit by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Profit (INR)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 24. CROP-WISE AVERAGE YIELD VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    crop_analysis.index,
    crop_analysis["Yield_Tonnes_Ha"]
)

plt.title("Average Yield by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Yield (Tonnes/Ha)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 25. SEASONAL ENVIRONMENTAL CONDITIONS ANALYSIS
# ============================================================

season_environment = df.groupby("Season").agg({
    "Rainfall_mm": "mean",
    "Avg_Temperature_C": "mean",
    "Humidity_pct": "mean",
    "Sunlight_Hours_Day": "mean",
    "Yield_Tonnes_Ha": "mean",
    "Profit_INR": "mean"
}).round(2)

print("\n========================================")
print("SEASONAL ENVIRONMENTAL CONDITIONS")
print("========================================")

print(season_environment)


# ============================================================
# 26. KEY FINDINGS AND EVIDENCE-BASED INSIGHTS
# ============================================================

print("\n========================================")
print("KEY FINDINGS AND EVIDENCE-BASED INSIGHTS")
print("========================================")

print("\n1. SEASONAL PERFORMANCE")
print("Best Season by Profit:", best_season_profit)
print(
    "Average Profit:",
    season_analysis.loc[best_season_profit, "Profit_INR"]
)
print("Highest Yield Season:", best_season_yield)
print(
    "Average Yield:",
    season_analysis.loc[best_season_yield, "Yield_Tonnes_Ha"]
)

print("\n2. CROP PERFORMANCE")
print("Most Profitable Crop:", best_profit_crop)
print(
    "Average Profit:",
    crop_analysis.loc[best_profit_crop, "Profit_INR"]
)
print("Highest Yield Crop:", best_yield_crop)
print(
    "Average Yield:",
    crop_analysis.loc[best_yield_crop, "Yield_Tonnes_Ha"]
)

print("\n3. STATE PERFORMANCE")
print("Most Profitable State:", best_state_profit)
print(
    "Average Profit:",
    state_analysis.loc[best_state_profit, "Profit_INR"]
)
print("Highest Production State:", best_state_production)
print(
    "Total Production:",
    state_analysis.loc[
        best_state_production,
        "Production_Tonnes"
    ]
)

print("\n4. IRRIGATION PERFORMANCE")
print(
    "Highest Yield Irrigation Method:",
    best_irrigation_yield
)
print(
    "Average Yield:",
    irrigation_analysis.loc[
        best_irrigation_yield,
        "Yield_Tonnes_Ha"
    ]
)
print(
    "Most Water-Efficient Irrigation Method:",
    best_irrigation_water
)
print(
    "Water Efficiency:",
    irrigation_analysis.loc[
        best_irrigation_water,
        "Water_Efficiency_t_per_1000m3"
    ]
)

print("\n5. ENVIRONMENTAL CONDITIONS")
print(
    "Rainfall-Yield Correlation:",
    weather_analysis.loc[
        "Rainfall_mm",
        "Yield_Tonnes_Ha"
    ]
)
print(
    "Temperature-Yield Correlation:",
    weather_analysis.loc[
        "Avg_Temperature_C",
        "Yield_Tonnes_Ha"
    ]
)
print(
    "Humidity-Yield Correlation:",
    weather_analysis.loc[
        "Humidity_pct",
        "Yield_Tonnes_Ha"
    ]
)

print("\n6. SOIL AND NUTRIENTS")
print(
    "Soil Moisture-Yield Correlation:",
    soil_analysis.loc[
        "Soil_Moisture_pct",
        "Yield_Tonnes_Ha"
    ]
)
print(
    "Nitrogen-Yield Correlation:",
    soil_analysis.loc[
        "Nitrogen_kg_ha",
        "Yield_Tonnes_Ha"
    ]
)
print(
    "Phosphorus-Yield Correlation:",
    soil_analysis.loc[
        "Phosphorus_kg_ha",
        "Yield_Tonnes_Ha"
    ]
)
print(
    "Potassium-Yield Correlation:",
    soil_analysis.loc[
        "Potassium_kg_ha",
        "Yield_Tonnes_Ha"
    ]
)

print("\n7. ECONOMIC RELATIONSHIPS")
print(
    "Revenue-Profit Correlation:",
    farm_area_analysis.loc[
        "Revenue_INR",
        "Profit_INR"
    ]
)
print(
    "Market Price-Profit Correlation:",
    market_analysis.loc[
        "Market_Price_INR_Tonne",
        "Profit_INR"
    ]
)

print("\n8. WATER USAGE")
print("Most Water-Efficient Method:", best_water_method)
print(
    "Water Efficiency:",
    water_analysis.loc[
        best_water_method,
        "Water_Efficiency_t_per_1000m3"
    ]
)

print("\n========================================")
print("DATA ANALYSIS COMPLETED SUCCESSFULLY")
print("========================================")