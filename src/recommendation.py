from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(dataset_dir):
    """lista as imagens do dataset em ordem previsivel."""
    dataset_path = Path(dataset_dir)

    return sorted(
        path
        for path in dataset_path.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_feature_extractor():
    """carrega a mobilenetv2 sem a camada final de classificacao."""
    return MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )


def prepare_image(image_path, target_size=(224, 224)):
    """abre a imagem no formato esperado pela rede."""
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def extract_embedding(model, image_path):
    """extrai o vetor de caracteristicas de uma imagem."""
    prepared = prepare_image(image_path)
    embedding = model.predict(prepared, verbose=0)
    return embedding.flatten()


def build_embeddings(model, image_paths):
    """gera embeddings para todas as imagens encontradas."""
    embeddings = []

    for path in image_paths:
        embeddings.append(extract_embedding(model, path))

    return np.array(embeddings)


def recommend_similar(query_image, image_paths, embeddings, model, top_k=3):
    """retorna as imagens mais parecidas com a imagem de consulta."""
    query_embedding = extract_embedding(model, query_image).reshape(1, -1)
    scores = cosine_similarity(query_embedding, embeddings)[0]

    ranked_indexes = np.argsort(scores)[::-1]
    recommendations = []

    query_path = Path(query_image).resolve()

    for index in ranked_indexes:
        candidate_path = Path(image_paths[index]).resolve()

        if candidate_path == query_path:
            continue

        recommendations.append(
            {
                "path": image_paths[index],
                "score": float(scores[index]),
            }
        )

        if len(recommendations) == top_k:
            break

    return recommendations


def show_recommendations(query_image, recommendations):
    """mostra a imagem buscada e as recomendacoes em um grid simples."""
    total_images = len(recommendations) + 1
    plt.figure(figsize=(4 * total_images, 4))

    plt.subplot(1, total_images, 1)
    plt.imshow(Image.open(query_image))
    plt.title("imagem de entrada")
    plt.axis("off")

    for index, item in enumerate(recommendations, start=2):
        plt.subplot(1, total_images, index)
        plt.imshow(Image.open(item["path"]))
        plt.title(f"similaridade: {item['score']:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
