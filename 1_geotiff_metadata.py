import pandas as pd
import pathlib



# Recursive list of all geotiff of directory
dir = pathlib.Path("/data/databases/spot_proba_cloud/CESAR_SPOT_VEGETATION/geotiff")
geotiffs = list(dir.rglob("*.tif"))

# Create a dataframe with the path and the name of the geotiff
df = pd.DataFrame(geotiffs, columns=["geotiff_path"])
df["folder"] = df["geotiff_path"].apply(lambda x: x.parent.name)
df["name"] = df["geotiff_path"].apply(lambda x: x.name.split(".")[0])
df["date"] = df["ñ"].apply(lambda x: x.split("_")[1])

