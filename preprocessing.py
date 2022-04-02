from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import glob 
import cv2 as cv

# Retrieve current directory
cwd = os.getcwd() + '\\dataset\\'
root_dir = os.path.dirname(cwd)

# Normalize images
for data in os.listdir(cwd):
    directory = os.path.join(root_dir, data, '*')
    directory2 = glob.glob(directory)
    for img in directory2:
        print(img)
        # Read image
        pic = cv.imread(img)
        # Resize new image to 100x100
        pic = cv.resize(pic, (100,100))
        # Normalize images
        cv.normalize(pic, pic, 0, 225, cv.NORM_MINMAX)
        # Override iamges
        cv.imwrite(img, pic) 

# Used for data argumentation 
for data in os.listdir(cwd):
    directory = os.path.join(root_dir, data, '*')
    directory2 = glob.glob(directory)
    for img in directory2:
        # Rotate images of up to 50 degrees
        datagen = ImageDataGenerator(rotation_range=50, fill_mode='nearest')
        
        # Load original image
        img = load_img(img)
        # Convert original image to array
        x = img_to_array(img)
        # Covert array to 4D
        x = x.reshape((1,) + x.shape)
           
        i = 0
        # Generate and save 2 rotated images
        for batch in datagen.flow(x,save_to_dir=os.getcwd() + '\\dataset\\' + data + "\\", save_prefix='generated', save_format='jpg'):
            if i > 0:
                break
            i += 1