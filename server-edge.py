# face_service.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np, cv2, faiss, json, uvicorn
from utils import mtcnn_detect_align, arcface_embed, load_faiss_index

app = FastAPI()
git index, meta = load_faiss_index("faces.index", "faces_meta.json")  # meta[id] = {"folder": ...}

class RecognizeResponse(BaseModel):
    matched: bool
    folder: str
    similarity: float

@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(image: UploadFile = File(...)):
    buf = await image.read()
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

    face = mtcnn_detect_align(img)               # 112×112 RGB
    if face is None:
        return {"matched": False, "folder": "", "similarity": 0}

    vec = arcface_embed(face)                    # 512-D float32
    vec = np.ascontiguousarray(vec).reshape(1, -1)
    sim, idx = index.search(vec, 1)              # cosine distance in Faiss = 2-cos
    sim = 1 - sim[0][0] / 2                      # convert back to cosine similarity

    if sim >= 0.35:                              # match
        folder = meta[str(idx[0][0])]["folder"]
        return {"matched": True, "folder": folder, "similarity": sim}
    else:                                        # new person
        new_id = index.ntotal
        index.add(vec)
        folder = f"person_{new_id}"
        meta[str(new_id)] = {"folder": folder}
        index.write_index(faiss.IndexFlatIP(512), "faces.index")
        with open("faces_meta.json", "w") as f: json.dump(meta, f)
        return {"matched": False, "folder": folder, "similarity": sim}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)