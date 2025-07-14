from fastapi import FastAPI, File, UploadFile
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

app = FastAPI()
model = ocr_predictor(pretrained=True)

@app.get("/")
def read_root():
    return {"message": "Hello, world! FastAPI is running."}

@app.post("/ocr/")
async def ocr_image(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp_image.png", "wb") as f:
        f.write(contents)
    doc = DocumentFile.from_images("temp_image.png")
    result = model(doc)
    text = ""
    exported = result.export()
    for page in exported['pages']:
        for block in page.get('blocks', []):
            for line in block.get('lines', []):
                text += " ".join([word['value'] for word in line.get('words', [])]) + "\n"
    return {"text": text}
