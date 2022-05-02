import cv2
from sklearn.decomposition import PCA
import os
import glob 
from PIL import Image

cwd = os.getcwd() + '\\dataset\\'
root_dir = os.path.dirname(cwd)
for data in os.listdir(cwd):
    directory = os.path.join(root_dir, data, '*')
    directory2 = glob.glob(directory)
    for img in directory2:
        location = img[img.find(data):]
        img = cv2.imread(img)
        
        red, green, blue = cv2.split(img) 
        pca = PCA(n_components=0.95)

        red_transformed = pca.fit_transform(red)
        red_inverted = pca.inverse_transform(red_transformed)
        
        green_transformed = pca.fit_transform(green)
        green_inverted = pca.inverse_transform(green_transformed)
        
        blue_transformed = pca.fit_transform(blue)
        blue_inverted = pca.inverse_transform(blue_transformed)

        img_compressed = (cv2.merge((red_inverted, green_inverted, blue_inverted)))

        cv2.imwrite('PCA\\' + location, img_compressed)