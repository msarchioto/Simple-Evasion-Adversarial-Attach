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
import argparse

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

def save_image(img, filename, size=1.2):
    if isinstance(img, np.ndarray):
        # Convert numpy array to PIL Image
        img = Image.fromarray((img * 255).astype(np.uint8))
    
    if size > 0:
        # Resize if size is specified
        current_size = img.size
        new_size = tuple(int(dim * size) for dim in current_size)
        img = img.resize(new_size)
    
    # Create output directory if it doesn't exist
    os.makedirs('output_images', exist_ok=True)
    output_path = os.path.join('output_images', filename)
    img.save(output_path)
    print(f"Saved image to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial attack on images')
    parser.add_argument('--image', type=str, default='images/airplane.jpg',
                      help='Path to input image')
    parser.add_argument('--target', type=str, default='ship',
                      choices=cifar10_class_names,
                      help='Target class for adversarial attack')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the image (without specifying target_size)
    img = load_img(args.image)
    save_image(img, 'original.png', size=-1)

    # Resize the image
    img_resized = img.resize((32, 32))
    save_image(img_resized, 'resized.png')
    
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
    target = np.array([cifar10_class_names.index(args.target)])  # Convert class name to index
    print(f"Target class index: {target}")    
    adv_image = attack.generate(x=img_data, y=target)
    adversarial_pred = predict(victim_model, adv_image)
    print(f"Adversarial prediction: {adversarial_pred}")

    # Save the image to adversarial_images folder
    altered_image = Image.fromarray((adv_image[0] * 255).astype(np.uint8))
    altered_image.save('adversarial_images/altered_adversarial_image.png')

if __name__ == "__main__":
    main()