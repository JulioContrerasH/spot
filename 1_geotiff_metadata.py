import pandas as pd
import pathlib

# Recursive list of all geotiff of directory
dir = pathlib.Path("/data/databases/spot_proba_cloud/CESAR_SPOT_VEGETATION/geotiff")
dir_tortillas = pathlib.Path("/data/databases/Julio/spot/tortillas")
geotiffs = list(dir.rglob("*.tif"))

# Create a dataframe with the path and the name of the geotiff
df = pd.DataFrame(geotiffs, columns=["geotiff_path"])
df["folder"] = df["geotiff_path"].apply(lambda x: x.parent.name)
df["name"] = df["geotiff_path"].apply(lambda x: x.name.split(".")[0])
df["id_sample"] = df["name"].str.split("_").str[-2:].str.join("_")
df.loc[df["id_sample"].str.split("_").str[-1] == "V003", "id_sample"] = "image"
df["date"] = df["geotiff_path"].apply(lambda x: x.parent.parent.name).str.split("_").str.join("-")
df["tortilla"] = dir_tortillas / (df["folder"] + ".tortilla")

# Write the dataframe to a csv file
df.to_csv("tables/metadata_geotiffs.csv", index=False)
