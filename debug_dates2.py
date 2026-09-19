import data_pipeline as dp
import pandas as pd

ft = dp.load_feature_table()
print("Types in Product-A reactor_date:", ft[ft["product"] == "Product-A"]["reactor_date"].map(type).value_counts())
print("Types in Product-B reactor_date:", ft[ft["product"] == "Product-B"]["reactor_date"].map(type).value_counts())
print("Sample Product-B rows:")
print(ft[ft["product"] == "Product-B"][["batch_no", "reactor_date", "reactor_ipqc_viscosity", "reactor_total_bct_min"]].head(10))
print("Product-B Viscosity dtype:", ft[ft["product"] == "Product-B"]["reactor_ipqc_viscosity"].dtype)
print("Product-B Viscosity values:")
print(ft[ft["product"] == "Product-B"]["reactor_ipqc_viscosity"].unique())
