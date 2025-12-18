import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import glob


def num1():
    # Load trained model
    model = load_model("animal_classifier_optimized.h5", compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Labels (classes in the same order as your dataset folders - ORDEN ALFABÉTICO)
    labels = ["catarina", "gato", "hormiga", "perro", "turtle"]  # Orden correcto según las carpetas

    # Target CNN input size
    target_w = 224
    target_h = 224

    def preprocess_image(img_path):
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError("Image not found:", img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize with padding to 21x28
        old_h, old_w = img.shape[:2]
        scale = min(target_w / old_w, target_h / old_h)
        new_w = int(old_w * scale)
        new_h = int(old_h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        # Normalize for the CNN (0–1)
        padded = padded.astype("float32") / 255.0

        # Add batch dimension → (1, 21, 28, 3)
        padded = np.expand_dims(padded, axis=0)

        return padded
    # ---- Test ----
    # Obtener todas las imágenes de la carpeta test
    test_folder = 'test'
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(test_folder, ext)))
    
    print(f"\n=== MODELO 1: animal_classifier_optimized.h5 ===")
    print(f"Procesando {len(all_images)} imágenes...\n")
    
    correct = 0
    total = len(all_images)
    
    for img_path in all_images:
        X = preprocess_image(img_path)
        prediction = model.predict(X, verbose=0)
        predicted_class = np.argmax(prediction)
        
        # Extraer nombre del archivo para mostrar
        img_name = os.path.basename(img_path)
        print(f"{img_name:20} -> {labels[predicted_class]}")
    
    print(f"\nTotal de imágenes procesadas: {total}")
    #img_path = 'test/cat.png'
    #img_path = 'test/dog.png'
    #img_path = 'test/ant.png'
    #img_path = 'test/catarina.png'
    #img_path = 'test/p2.png'
    
  
    #X = preprocess_image(img_path)

    #prediction = model.predict(X)
    #predicted_class = np.argmax(prediction)

    #print("Predicted class:", predicted_class)
    #print("Label:", labels[predicted_class])

    # Show the image
    #plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    #plt.title("Es un: " + labels[predicted_class])
    #plt.axis("off")
    #plt.show()

def num2():
   
    print(f"\n=== MODELO 2: aanimal_classifier_optimized.h5 ===")
    
    # Cargar el modelo h5
    modelo_h5 = 'aanimal_classifier_optimized.h5'
    
    # Verificar si existe el modelo
    if not os.path.exists(modelo_h5):
        print(f"Modelo {modelo_h5} no encontrado. Saltando num2().")
        return
        
    riesgo_model = load_model(modelo_h5, compile=False)
    riesgo_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Obtener todas las imágenes de la carpeta test
    test_folder = 'test'
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(test_folder, ext)))
    
    print(f"Procesando {len(all_images)} imágenes...\n")
    
    images = []
    filenames = []
    
    for filepath in all_images:
        image = plt.imread(filepath)
        images.append(image)
        filenames.append(filepath)

    X = np.array(images, dtype=np.uint8)  # Convierto de lista a numpy
    test_X = X.astype('float32')
    test_X = test_X / 255.

    predicted_classes = riesgo_model.predict(test_X, verbose=0)

    # Asegúrate de tener una lista de etiquetas o categorías en 'sriesgos'
    sriesgos = ["catarina", "gato", "hormiga", "perro", "turtle"]  # Orden correcto según las carpetas

    for i, img_tagged in enumerate(predicted_classes):
        img_name = os.path.basename(filenames[i])
        print(f"{img_name:20} -> {sriesgos[np.argmax(img_tagged)]}")
    
    print(f"\nTotal de imágenes procesadas: {len(filenames)}")
    
    #images = []
    # AQUI ESPECIFICAMOS UNAS IMAGENES
    #filenames = ['/home/panque/repos/IA/Eigenface/animals/test/cat3.jpg','/home/panque/repos/IA/Eigenface/animals/test/dog.jpg',
    #             '/home/panque/repos/IA/Eigenface/animals/test/cat2.jpg','/home/panque/repos/IA/Eigenface/animals/test/ant.jpg',
    #             '/home/panque/repos/IA/Eigenface/animals/test/ladybug.jpg','/home/panque/repos/IA/Eigenface/animals/test/turtle.jpg']

    #for filepath in filenames:
    #    image = plt.imread(filepath)
    #    #image_resized = resize(image, (21, 28), anti_aliasing=True, clip=False, preserve_range=True)
    #    images.append(image)

    #X = np.array(images, dtype=np.uint8)  # Convierto de lista a numpy
    #test_X = X.astype('float32')
    #test_X = test_X / 255.

    #predicted_classes = riesgo_model.predict(test_X)

    # Asegúrate de tener una lista de etiquetas o categorías en 'sriesgos'
    #sriesgos = ["catarina", "gato", "hormiga", "perro", "turtle"]  # Orden correcto según las carpetas

    #for i, img_tagged in enumerate(predicted_classes):
    #    print(filenames[i], sriesgos[np.argmax(img_tagged)])

if __name__ == "__main__":
    num1()
    num2()