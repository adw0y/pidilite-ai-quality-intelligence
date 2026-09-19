import data_pipeline as dp
import pandas as pd

ft = dp.load_feature_table()
print("Unique products:", ft["product"].unique())
print("\nProduct-A dates:")
print(ft[ft["product"] == "Product-A"]["reactor_date"].head())
print("\nProduct-B dates:")
print(ft[ft["product"] == "Product-B"]["reactor_date"].head())
print("\nProduct-B null count in reactor_date:", ft[ft["product"] == "Product-B"]["reactor_date"].isna().sum())
print("\nProduct-B null count in viscosity:", ft[ft["product"] == "Product-B"]["reactor_ipqc_viscosity"].isna().sum())
print("\nProduct-B null count in total_bct:", ft[ft["product"] == "Product-B"]["reactor_total_bct_min"].isna().sum())
