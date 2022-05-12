import cv2
from sklearn.decomposition import PCA
import os
import glob 
from PIL import Image

# Locate dataset folder
cwd = os.getcwd() + '\\dataset\\'
dataset = os.path.dirname(cwd)

# Location for PCA folder
pca_folder = os.getcwd() + '\\PCA_dataset\\'

# If PCA folder does not exist create PCA folder
if not os.path.exists(pca_folder):
    os.makedirs(pca_folder)

# Loop through dataset folder
for data in os.listdir(cwd):
    # Retrieve the folders within dataset
    directory = os.path.join(dataset, data, '*')
    directory2 = glob.glob(directory)

    # Location to check for folders within PCA_folder containing the labeled images
    pca_weather_folders = pca_folder + data + "\\"

    # Check if location for labeled images exist
    if not os.path.exists(pca_weather_folders):
        os.makedirs(pca_weather_folders)

    # Loop through images within folder
    for img in directory2:
        # find the location of the images
        location = img[img.find(data):]
        # Open images
        img = cv2.imread(img)
        
        # Retrieve the RGB array of image
        red, green, blue = cv2.split(img)
        # Create PCA with 95% variance
        pca = PCA(n_components=0.95)

        # Transform and invert red 
        red_transformed = pca.fit_transform(red)
        red_inverted = pca.inverse_transform(red_transformed)
        
        # Transform and invert green
        green_transformed = pca.fit_transform(green)
        green_inverted = pca.inverse_transform(green_transformed)
        
        # Transform and invert blue
        blue_transformed = pca.fit_transform(blue)
        blue_inverted = pca.inverse_transform(blue_transformed)

        # Merge the red green and blue
        img_compressed = (cv2.merge((red_inverted, green_inverted, blue_inverted)))
        
        # Save image
        cv2.imwrite('PCA_dataset\\' + location, img_compressed)