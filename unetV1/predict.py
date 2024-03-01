
import tensorflow as tf
import os
import  cv2
import  numpy as np
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef,iou


""" Global parameters """
H = 256
W = 256
path = "/home/mohamed/Desktop/Skin-Cancer-detection/BDD_ISIC_2019_NORM/NAEVUS"
result_path = "Mask-1/"

""" Load the model """
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
    model = tf.keras.models.load_model("files/modelunetseg.h5")


if not os.path.exists(result_path):
    os.makedirs(result_path)
i = 0
for filename in sorted(os.listdir(path)):
    if filename.endswith(".JPG"):
        img = cv2.imread(os.path.join(path, filename))

        image = cv2.resize(img, (W, H))       ## [H, w, 3]
        x = image/255.0                         ## [H, w, 3]
        x = np.expand_dims(x, axis=0)           ## [1, H, w, 3]
        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)

        y_pred = np.expand_dims(y_pred, axis=-1)
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
        y_pred = y_pred * 255
        line = np.ones((H, 10, 3)) * 255
        cv2.imwrite(f"/home/mohamed/Desktop/Skin-Cancer-detection/BDD_ISIC_2019_NORM/NAEVUSMASK/image{i}.jpg", y_pred)
        i = i+1



