import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import warnings
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from PIL import Image
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import load_img, img_to_array

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
tf.compat.v1.disable_eager_execution()

# Set the environment variable to limit TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses most logs

# Initialize TensorFlow and check it doesn't output other logs
tf.get_logger().setLevel('ERROR')

cifar10_class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def predict(model,img):
    pred = model.predict(img)
    label = np.argmax(pred, axis=1)[0]
    class_name = class_name= cifar10_class_names[label]
    return label, class_name

def show_image(img, size = 1.2):
    if size>0:
        plt.figure(figsize=(size, size), dpi=80)
    plt.imshow(img)
    plt.axis('off')   

def main():
    # Load the image (without specifying target_size)
    img = load_img('images/airplane.jpg')
    show_image(img, size=-1)

    # Resize the image
    img_resized = img.resize((32, 32))
    show_image(img_resized)
    # Convert the image back to a NumPy array for use with Keras
    img_array = img_to_array(img_resized)
    # Normalize the pixel values
    img_data = np.array([img_array.astype('float32') / 255.0])

    # Load the pre-trained victim model
    victim_model =  tf.keras.models.load_model('models/simple-cifar10.h5',compile=False)

    original_pred = predict(victim_model, img_data)
    print(f"Original prediction: {original_pred}")

    # convert keras model to ART model
    classifier = KerasClassifier(model=victim_model, clip_values=(0, 1), use_logits=False)
    attack = FastGradientMethod(estimator=classifier, eps=0.01, targeted=True)
    target = np.array([8])  # Targeting the 'ship' class
    adv_image = attack.generate(x=img_data, y=target)
    adversarial_pred = predict(victim_model, adv_image)
    print(f"Adversarial prediction: {adversarial_pred}")

    show_image(adv_image[0])

    # Save the image
    altered_image = Image.fromarray((adv_image[0] * 255).astype(np.uint8))
    altered_image.save('adversarial_images/altered_adversarial_image.png')

if __name__ == "__main__":
    main()