import streamlit as st
import os
import glob
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.cluster import KMeans
import numpy as np
from streamlit_drawable_canvas import st_canvas
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configuration
DATA_DIR = "data/raw"
ANNOTATION_FILE = "data/processed/annotations.json"
NUM_CLUSTERS = 5
IMG_SIZE = (224, 224)

# --- Helper Functions ---

def ensure_directories():
    """Ensure necessary directories exist."""
    os.makedirs(os.path.dirname(ANNOTATION_FILE), exist_ok=True)
    if not os.path.exists(ANNOTATION_FILE):
        with open(ANNOTATION_FILE, 'w') as f:
            json.dump({}, f)

@st.cache_resource
def get_model():
    """Load pre-trained ResNet18 for feature extraction."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove FC layer
    model.eval()
    return model

@st.cache_data
def load_images(data_dir: str) -> List[str]:
    """Load image paths from the directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    return sorted(list(set(image_paths)))

@st.cache_data
def get_features_for_images(image_paths: List[str]) -> np.ndarray:
    """Extract CNN features for a list of images. Cached to avoid re-computation."""
    model = get_model()
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0)
            with torch.no_grad():
                feat = model(img_t).flatten().numpy()
            features.append(feat)
        except Exception as e:
            features.append(np.zeros(512))

    return np.array(features)

def cluster_images(features: np.ndarray, n_clusters: int = 5) -> List[int]:
    """Cluster images using KMeans."""
    if len(features) < n_clusters:
        n_clusters = len(features)
    if n_clusters == 0:
        return []
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(features)

def save_annotation(img_path: str, data: Dict[str, Any]):
    """Save annotation to JSON file."""
    ensure_directories()
    try:
        with open(ANNOTATION_FILE, 'r') as f:
            annotations = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        annotations = {}

    # Use relative path as key to avoid issues with absolute paths changing
    key = os.path.relpath(img_path, start=os.getcwd())

    if key not in annotations:
        annotations[key] = {}

    annotations[key].update(data)
    annotations[key]["last_updated"] = datetime.now().isoformat()

    with open(ANNOTATION_FILE, 'w') as f:
        json.dump(annotations, f, indent=4)

def get_annotation(img_path: str) -> Dict[str, Any]:
    """Retrieve annotation for an image."""
    ensure_directories()
    try:
        with open(ANNOTATION_FILE, 'r') as f:
            annotations = json.load(f)
        key = os.path.relpath(img_path, start=os.getcwd())
        return annotations.get(key, {})
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

# --- App Interface ---

def main():
    st.set_page_config(page_title="Projective Drawings Annotation", layout="wide")
    st.title("Human-in-the-Loop: Visual Knowledge Graph Builder")

    ensure_directories()

    # Navigation
    mode = st.sidebar.radio("Workflow Mode", ["Cluster Labeling", "Decomposition (Parts)"])

    # Load Data
    if not os.path.exists(DATA_DIR):
        st.warning(f"Data directory `{DATA_DIR}` does not exist. Creating it...")
        os.makedirs(DATA_DIR, exist_ok=True)

    image_paths = load_images(DATA_DIR)

    if not image_paths:
        st.warning(f"No images found in `{DATA_DIR}`. Please add images to start.")
        st.info("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
        return

    if mode == "Cluster Labeling":
        st.header("Cluster View: Unlabeled Images")
        st.markdown("Group images by visual similarity to speed up labeling.")

        with st.spinner("Processing images..."):
            features = get_features_for_images(image_paths)

            if len(features) > 0:
                labels = cluster_images(features, n_clusters=min(NUM_CLUSTERS, len(features)))

                # Organize by cluster
                clusters: Dict[int, List[str]] = {}
                for idx, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(image_paths[idx])

                # Display Clusters
                for cluster_id in sorted(clusters.keys()):
                    with st.expander(f"Cluster {cluster_id} ({len(clusters[cluster_id])} images)", expanded=True):
                        cols = st.columns(5)
                        for i, img_path in enumerate(clusters[cluster_id]):
                            with cols[i % 5]:
                                st.image(img_path, use_container_width=True)
                                st.caption(os.path.basename(img_path))

                                # Quick Label Form per image
                                current_data = get_annotation(img_path)
                                with st.popover("Label"):
                                    st.text_input("Child's Title", value=current_data.get("title", ""), key=f"title_{cluster_id}_{i}")
                                    st.text_input("Expert Label", value=current_data.get("label", ""), key=f"label_{cluster_id}_{i}")

                                    if st.button("Save", key=f"save_{cluster_id}_{i}"):
                                        # Retrieve values from session state
                                        title_val = st.session_state[f"title_{cluster_id}_{i}"]
                                        label_val = st.session_state[f"label_{cluster_id}_{i}"]
                                        save_annotation(img_path, {"title": title_val, "label": label_val})
                                        st.success("Saved!")
            else:
                st.error("Failed to extract features.")

    elif mode == "Decomposition (Parts)":
        st.header("Decomposition Mode: Sub-part Annotation")
        st.markdown("Draw bounding boxes to identify and label specific parts of the drawing.")

        selected_img_path = st.selectbox("Select Image", image_paths)

        if selected_img_path:
            current_data = get_annotation(selected_img_path)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Draw Bounding Boxes")

                img = Image.open(selected_img_path)
                width, height = img.size

                # Set canvas size
                canvas_width = 600
                # Calculate scale factor for saving coordinates relative to original image
                scale_factor = width / canvas_width

                canvas_height = int(height * (canvas_width / width))

                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#FF0000",
                    background_image=img,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect",
                    key="canvas",
                )

            with col2:
                st.subheader("Metadata & Annotations")

                # Input form for the whole image
                st.markdown("### Global Labels")
                child_title = st.text_input("Child's Title", value=current_data.get("title", ""), key="decomp_title")
                expert_label = st.text_input("Expert Label", value=current_data.get("label", ""), key="decomp_label")

                parts_data = []
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if len(objects) > 0:
                        st.markdown("### Detected Parts")
                        st.info("Label each detected part below:")

                        for idx, obj in enumerate(objects):
                            with st.container():
                                col_a, col_b = st.columns([1, 3])
                                with col_a:
                                    st.text(f"Part {idx+1}")
                                with col_b:
                                    label_input = st.text_input(
                                        f"Label for Box {idx+1}",
                                        key=f"part_label_{idx}",
                                        placeholder="e.g., sharp teeth"
                                    )

                                parts_data.append({
                                    "id": idx,
                                    "box": {
                                        "left": int(obj["left"] * scale_factor),
                                        "top": int(obj["top"] * scale_factor),
                                        "width": int(obj["width"] * scale_factor),
                                        "height": int(obj["height"] * scale_factor)
                                    },
                                    "label": label_input
                                })

                        st.divider()
                        st.info(f"{len(objects)} parts detected.")

                if st.button("Save All Annotations"):
                    data_to_save = {
                        "title": child_title,
                        "label": expert_label,
                        "parts": parts_data
                    }
                    save_annotation(selected_img_path, data_to_save)
                    st.success(f"Saved metadata for {os.path.basename(selected_img_path)}!")

if __name__ == "__main__":
    main()
