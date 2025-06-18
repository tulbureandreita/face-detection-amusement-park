import os
import cv2
import numpy as np
import insightface
import faiss

# Init Actiface app with SCRFD + ArcFace
model = insightface.app.FaceAnalysis(name='antelopev2')  # SCRFD + ArcFace
model.prepare(ctx_id=0)  # 0 = GPU if available; -1 for CPU

print("Loaded models:", model.models.keys())  # Debug print

# Folder with known people
KNOWN_DIR = "dev-images"

# Build index and labels
embeddings = []
names = []

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    faces = model.get(img)
    if not faces:
        raise Exception(f"No face found in {image_path}")
    return faces[0].embedding

# Load and embed known faces
for filename in os.listdir(KNOWN_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        try:
            emb = extract_embedding(os.path.join(KNOWN_DIR, filename))
            embeddings.append(emb)
            names.append(name)
            print(f"Embedded {name}")
        except Exception as e:
            print(e)

embeddings = np.array(embeddings).astype("float32")

# Create and train Faiss index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Recognize test face
test_img = cv2.imread("test.jpg")
faces = model.get(test_img)

if not faces:
    print("No face found in test image.")
else:
    test_embedding = faces[0].embedding.astype("float32")
    D, I = index.search(np.expand_dims(test_embedding, axis=0), k=1)
    distance = D[0][0]
    match_idx = I[0][0]
    matched_name = names[match_idx]

    print(f"Match: {matched_name} (distance={distance:.3f})")
