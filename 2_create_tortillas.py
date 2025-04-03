import pandas as pd
import rasterio as rio
import datetime
import tacotoolbox
from concurrent.futures import ProcessPoolExecutor, as_completed
import ee
import tacoreader
import matplotlib.pyplot as plt

# Initialize Google Earth Engine
ee.Authenticate()
ee.Initialize(project="ee-contrerasnetk")

# Read the metadata table
table = pd.read_csv("tables/metadata_geotiffs.csv")

# Group the table by the 'folder' column
groups = table.groupby("folder")
len_group = len(groups)

# -----------------------------------------------------------------------------
# 1) Create samples as tortillas
# -----------------------------------------------------------------------------
def create_sample_from_group(group_tuple: tuple) -> int:
    """
    Creates a Sample object for a given group (folder_name, group).
    
    Args:
        group_tuple (tuple): A tuple containing folder_name (str) and 
                              group (pandas DataFrame).
    
    Returns:
        int: Returns 1 to indicate that the group has been processed.
    """
    folder_name, group, x = group_tuple

    # Extract the rows corresponding to different types of masks
    image = group[group["id_sample"] == "image"].iloc[0]
    initial_mask = group[group["id_sample"] == "initial_mask"].iloc[0]
    cloud_mask = group[group["id_sample"] == "cloud_mask"].iloc[0]
    mask_cnn = group[group["id_sample"] == "mask_cnn"].iloc[0]
   
    # Create Sample for each image and mask
    sample_image = create_sample(image, "image")
    sample_initial_mask = create_sample(initial_mask, "initial_mask")
    sample_cloud_mask = create_sample(cloud_mask, "cloud_mask")
    sample_mask_cnn = create_sample(mask_cnn, "cnn_mask")

    # Combine all samples
    samples = tacotoolbox.tortilla.datamodel.Samples(
        samples=[sample_image, sample_initial_mask, sample_cloud_mask, sample_mask_cnn]
    )

    # Create the tortilla
    tacotoolbox.tortilla.create(samples, image["tortilla"], quiet=True)

    print(f"Processed group {x+1}/{len_group}")
    return 1


def create_sample(row: pd.Series, sample_type: str) -> tacotoolbox.tortilla.datamodel.Sample:
    """
    Helper function to create a Sample from a given row.

    Args:
        row (pd.Series): A row from the dataframe containing the sample information.
        sample_type (str): The type of sample, e.g., "image", "initial_mask".

    Returns:
        Sample: The created Sample object.
    """
    with rio.open(row["geotiff_path"]) as src:
        profile = src.profile
        sample = tacotoolbox.tortilla.datamodel.Sample(
            id=sample_type,
            path=row["geotiff_path"],
            file_format="GTiff",
            data_split="train",
            stac_data={
                "crs": "EPSG:" + str(profile["crs"].to_epsg()),
                "geotransform": profile["transform"].to_gdal(),
                "raster_shape": (profile["height"], profile["width"]),
                "time_start": datetime.datetime.strptime(row.date, '%Y-%m-%d'),
                "time_end": datetime.datetime.strptime(row.date, '%Y-%m-%d')
            },
            roi=row["folder"]  # Use folder name as the region of interest
        )
    return sample

processed = 0  # Count processed groups
n_workers = 4  # Adjust the number of workers based on your resources

# Using ProcessPoolExecutor to process groups in parallel
with ProcessPoolExecutor(max_workers=n_workers) as executor:
    # Create a list of futures for each group
    future_to_group = {
        executor.submit(create_sample_from_group, (folder_name, group, x)): folder_name
        for x, (folder_name, group) in enumerate(groups)
    }
    
    # Collect results as they finish
    for future in as_completed(future_to_group):
        try:
            result = future.result() 
            processed += result
        except Exception as e:
            print(f"Error processing group: {e}")

# -----------------------------------------------------------------------------
# 2) Create the general tortilla
# -----------------------------------------------------------------------------
# Filter out the first row for each folder in the table
table_tortillas = table.groupby("folder").head(1)
table_tortillas = table_tortillas[table_tortillas["folder"].isin(
    ["V1KRNP____19990110F165_V003", "V1KRNP____19990110F166_V003", "V1KRNP____19990110F171_V003"]
)] # This a just test, you can remove it

# Define the function to process each row (in parallel)
def create_sample_from_row(row_tuple: tuple) -> tacotoolbox.tortilla.datamodel.Sample:
    """
    Creates a Sample for each row in the DataFrame.

    Args:
        row_tuple (tuple): A tuple containing index (int) and row (pd.Series).

    Returns:
        Sample: The created Sample object.
    """
    index, row = row_tuple
    
    # Load tortilla data
    sample_data = tacoreader.load(row["tortilla"]).iloc[0]
    
    # Create Sample object
    sample_tortilla = tacotoolbox.tortilla.datamodel.Sample(
        id=row["folder"],
        path=row["tortilla"],
        file_format="TORTILLA",
        stac_data={
            "crs": sample_data["stac:crs"],
            "geotransform": sample_data["stac:geotransform"],
            "raster_shape": sample_data["stac:raster_shape"],
            "centroid": sample_data["stac:centroid"],
            "time_start": sample_data["stac:time_start"],
            "time_end": sample_data["stac:time_end"],
        }
    )    
    return sample_tortilla

# List to store all Samples
sample_tortillas = []

n_workers = 4
# Parallelize the row processing
with ProcessPoolExecutor(max_workers=n_workers) as executor:
    future_to_row = {
        executor.submit(create_sample_from_row, (index, row)): index
        for index, row in table_tortillas.iterrows()
    }
    
    # Collect results
    for future in as_completed(future_to_row):
        try:
            sample_tortilla = future.result()
            sample_tortillas.append(sample_tortilla)
        except Exception as e:
            print(f"Error processing the row: {e}")

# Create Samples object
samples = tacotoolbox.tortilla.datamodel.Samples(samples=sample_tortillas)

# Add RAI metadata to footer
samples_obj = samples.include_rai_metadata(
    sample_footprint=5120, 
    cache=False,  
    quiet=False  
)

# Finally, create the tortilla
tacotoolbox.tortilla.create(samples_obj, "/data/databases/Julio/spot/tortilla/spot.tortilla", quiet=True)

# -----------------------------------------------------------------------------
# 3) Load and display data
# -----------------------------------------------------------------------------
dataset = tacoreader.load("/data/databases/Julio/spot/tortilla/spot.tortilla")
row = dataset.read(0)

# Retrieve the data
image, initial_mask, cloud_mask, cnn_mask = row.read(0), row.read(1), row.read(2), row.read(3)

# Open the images for display
with rio.open(image) as src, rio.open(initial_mask) as src2, rio.open(cloud_mask) as src3, rio.open(cnn_mask) as src4:
    data = src.read([2, 3, 4])
    data_initial_mask = src2.read(1)
    data_cloud_mask = src3.read(1)
    data_cnn_mask = src4.read(1)

# Display the images
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# SPOT Image
ax[0, 0].imshow(data.transpose(1, 2, 0) * 4)
ax[0, 0].axis('off')
ax[0, 0].set_title('SPOT', fontsize=16)

# Initial Mask
ax[0, 1].imshow(data_initial_mask, cmap='gray')
ax[0, 1].axis('off')
ax[0, 1].set_title('Initial Mask', fontsize=16)

# Cloud Mask
ax[1, 0].imshow(data_cloud_mask, cmap='gray')
ax[1, 0].axis('off')
ax[1, 0].set_title('Cloud Mask', fontsize=16)

# CNN Mask
ax[1, 1].imshow(data_cnn_mask, cmap='gray')
ax[1, 1].axis('off')
ax[1, 1].set_title('CNN Mask', fontsize=16)

# Adjust spacing between images
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Save the image with minimal borders
plt.savefig("image.png", bbox_inches='tight', pad_inches=0.05)