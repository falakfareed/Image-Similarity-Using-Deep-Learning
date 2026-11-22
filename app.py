import os
import pickle
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from tensorflow.keras.models import load_model, Model
from sklearn.neighbors import NearestNeighbors


# =====================================
# SETTINGS
# =====================================

MODEL_PATH = "MobileNetV2_model.keras"
DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)
# Naya naam: purani image_database.pkl (galat scale wali) ignore ho jaye gi
DATABASE_FILE = "image_database_v2.pkl"

CLASSES = [
    "bike",
    "cars",
    "cat",
    "dog",
    "flowers",
    "horse",
    "riders"
]

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.95
SIMILARITY_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.30       # top-1 aur top-2 probability ka farq
AGREEMENT_THRESHOLD = 0.60    # top-5 neighbours me predicted class ka ratio


# =====================================
# STREAMLIT PAGE
# =====================================

st.set_page_config(
    page_title="Image Similarity Finder",
    page_icon="🖼️",
    layout="wide"
)


# =====================================
# CSS DESIGN
# =====================================

st.markdown(""" <style> .stApp { background: linear-gradient( 135deg, #eef2ff, #f8fafc, #fdf2f8 ); color: #1e293b; } .block-container { max-width: 1200px; padding-top: 2rem; } h1 { text-align: center; color: #4f46e5; font-weight: 800; } h2, h3 { color: #312e81; } [data-testid="stFileUploader"] section { background: white; border: 2px dashed #818cf8; border-radius: 15px; padding: 20px; } [data-testid="stImage"] img { border-radius: 14px; } [data-testid="column"] { background: white; padding: 12px; border-radius: 14px; box-shadow: 0 4px 14px rgba(0,0,0,0.08); } [data-testid="stAlert"] { border-radius: 12px; } </style> """, unsafe_allow_html=True)


st.title("🖼️ Image Similarity Finder")

st.write(
    "Upload an image to find similar images "
    "using MobileNetV2."
)


# =====================================
# LOAD MOBILENET MODEL
# =====================================

@st.cache_resource
def load_mobilenet_model():

    model = load_model(MODEL_PATH)

    embedding_model = Model(
        inputs=model.input,
        outputs=model.get_layer("embedding").output
    )

    return model, embedding_model


if not os.path.exists(MODEL_PATH):

    st.error("MobileNet model file not found!")

    st.stop()


model, embedding_model = load_mobilenet_model()


# =====================================
# GET IMAGE EMBEDDING
# =====================================

def get_embedding(image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Notebook me training /255.0 se hui thi, isliye yahan bhi wahi
    image_array = image_array / 255.0

    embedding = embedding_model.predict(
        image_array,
        verbose=0
    )[0]

    norm = np.linalg.norm(embedding)

    if norm != 0:

        embedding = embedding / norm

    return embedding


# =====================================
# FILE HASH (duplicate images pehchanne ke liye)
# =====================================

@st.cache_data
def get_file_hash(path):

    try:

        with open(path, "rb") as file:

            return hashlib.md5(file.read()).hexdigest()

    except Exception:

        return None


# =====================================
# CREATE IMAGE DATABASE
# =====================================

@st.cache_resource
def create_database():

    # Load existing database

    if os.path.exists(DATABASE_FILE):

        with open(DATABASE_FILE, "rb") as file:

            paths, labels, embeddings = pickle.load(file)

        if len(embeddings) > 0:

            return paths, labels, embeddings

    paths = []
    labels = []
    embeddings = []

    # Read all 7 classes

    for class_name in CLASSES:

        class_path = os.path.join(
            DATASET_PATH,
            class_name
        )

        if not os.path.exists(class_path):

            continue

        for file_name in os.listdir(class_path):

            file_path = os.path.join(
                class_path,
                file_name
            )

            try:

                image = Image.open(
                    file_path
                ).convert("RGB")

                embedding = get_embedding(image)

                paths.append(file_path)

                labels.append(class_name)

                embeddings.append(embedding)

            except Exception:

                continue

    if len(embeddings) == 0:

        return None, None, None

    embeddings = np.array(embeddings)

    # Save database

    with open(DATABASE_FILE, "wb") as file:

        pickle.dump(
            (paths, labels, embeddings),
            file
        )

    return paths, labels, embeddings


# =====================================
# PREPARE DATABASE
# =====================================

with st.spinner("Preparing image database..."):

    image_paths, image_labels, database_embeddings = (
        create_database()
    )


# =====================================
# CHECK DATABASE
# =====================================

if database_embeddings is None:

    st.error(
        "Dataset images were not found. "
        "Please check your dataset folder."
    )

    st.stop()


st.success(
    f"Database Ready: {len(image_paths)} images"
)


# =====================================
# UPLOAD QUERY IMAGE
# =====================================

uploaded_file = st.file_uploader(
    "Upload Query Image",
    type=["jpg", "jpeg", "png"]
)


# =====================================
# IMAGE SIMILARITY SEARCH
# =====================================

if uploaded_file is not None:

    try:

        query_image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error("Invalid image file!")

        st.stop()


    st.subheader("Query Image")

    st.image(
        query_image,
        width=300
    )


    # =================================
    # GENERATE QUERY EMBEDDING
    # =================================

    query_embedding = get_embedding(
        query_image
    )


    # =================================
    # IRRELEVANT IMAGE DETECTION
    # =================================

    st.subheader("Image Validation")

    # Prepare image for classifier

    query_array = query_image.resize(
        IMG_SIZE
    )

    query_array = np.array(
        query_array,
        dtype=np.float32
    )

    query_array = np.expand_dims(
        query_array,
        axis=0
    )

    query_array = query_array / 255.0

    # MobileNet classification

    predictions = model.predict(
        query_array,
        verbose=0
    )

    # Check classifier output

    if (
        predictions.ndim != 2
        or predictions.shape[1] != len(CLASSES)
    ):

        st.error(
            "Your model does not have a 7-class "
            "classification output. Check your model."
        )

        st.stop()


    # Get predicted class

    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_class = CLASSES[
        predicted_index
    ]

    confidence = float(
        predictions[0][predicted_index]
    )

    # Top-1 aur top-2 probability ka farq
    sorted_probs = np.sort(predictions[0])[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1])


    # =================================
    # FIND NEAREST DATABASE IMAGES
    # =================================

    check_knn = NearestNeighbors(
        n_neighbors=min(5, len(database_embeddings)),
        metric="cosine"
    )

    check_knn.fit(
        database_embeddings
    )

    check_distances, check_indices = (
        check_knn.kneighbors(
            [query_embedding]
        )
    )

    # Top-3 neighbours ki average similarity
    highest_similarity = float(
        np.mean(1 - check_distances[0][:3])
    )

    # Top-5 neighbours me kitne predicted class ke hain
    neighbour_labels = [
        image_labels[i] for i in check_indices[0]
    ]

    agreement = (
        neighbour_labels.count(predicted_class)
        / len(neighbour_labels)
    )


    # =================================
    # RELEVANT / IRRELEVANT CHECK
    # =================================

    if (
        confidence < CONFIDENCE_THRESHOLD
        or margin < MARGIN_THRESHOLD
        or highest_similarity < SIMILARITY_THRESHOLD
        or agreement < AGREEMENT_THRESHOLD
    ):

        st.error(
            "🚫 IRRELEVANT IMAGE DETECTED!"
        )

        st.warning(
            "This image does not belong to "
            "our dataset categories."
        )

        st.info(
            "Please upload an image from one "
            "of our 7 dataset classes."
        )

        st.write(
            f"Classification Confidence: "
            f"{confidence * 100:.2f}%"
        )

        st.write(
            f"Database Similarity: "
            f"{highest_similarity * 100:.2f}%"
        )

        st.stop()


    else:

        st.success(
            "✅ RELEVANT IMAGE DETECTED!"
        )

        st.write(
            f"Predicted Class: {predicted_class}"
        )

        st.write(
            f"Classification Confidence: "
            f"{confidence * 100:.2f}%"
        )

        st.write(
            f"Database Similarity: "
            f"{highest_similarity * 100:.2f}%"
        )


    # =================================
    # NUMBER OF SIMILAR IMAGES
    # =================================

    number_of_results = st.slider(
        "Number of similar images",
        min_value=1,
        max_value=min(10, len(image_paths)),
        value=min(5, len(image_paths))
    )


    # =================================
    # FIND SIMILAR IMAGES
    # =================================

    # Zyada neighbours lete hain taake duplicates hatane ke baad
    # bhi poore results mil saken
    knn = NearestNeighbors(
        n_neighbors=min(50, len(database_embeddings)),
        metric="cosine"
    )

    knn.fit(
        database_embeddings
    )

    distances, indices = knn.kneighbors(
        [query_embedding]
    )

    # Upload ki hui image aur dataset ki duplicate copies hata do
    seen_hashes = {
        hashlib.md5(uploaded_file.getvalue()).hexdigest()
    }

    unique_results = []

    for dist, idx in zip(distances[0], indices[0]):

        file_hash = get_file_hash(image_paths[idx])

        if file_hash is None or file_hash in seen_hashes:

            continue

        seen_hashes.add(file_hash)

        unique_results.append((dist, idx))

        if len(unique_results) == number_of_results:

            break


    st.subheader("Similar Images")

    if len(unique_results) == 0:

        st.info("No other similar images found.")

        st.stop()

    cols = st.columns(
        len(unique_results)
    )

    results = []


    for i, (distance, index) in enumerate(unique_results):

        similarity = (
            1 - distance
        ) * 100

        image_path = image_paths[index]

        label = image_labels[index]

        with cols[i]:

            st.image(
                image_path,
                use_container_width=True
            )

            st.write(
                f"*Class:* {label}"
            )

            st.write(
                f"*Similarity:* {similarity:.2f}%"
            )

        results.append({

            "Image": os.path.basename(image_path),

            "Class": label,

            "Similarity (%)": round(
                similarity,
                2
            )

        })


    # =================================
    # DISPLAY RESULTS TABLE
    # =================================

    st.subheader("Search Results")

    results_df = pd.DataFrame(
        results
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )


# =====================================
# END OF APPLICATION
# =====================================