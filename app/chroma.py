import chromadb
from io import BytesIO
from hashlib import sha3_256
import pypdf
import docx
import magic

CHUNK_SIZE = 512
OVERLAP_SIZE = 32


def read_pdf(file_path) -> str:
    pdf = pypdf.PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP_SIZE) -> list[str]:
    chunks = []
    # Simple character-based chunking
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size]
        if chunk:  # Ensure we don't add empty chunks
            chunks.append(chunk)

    return chunks


def detect_file_type(file: BytesIO):
    # Reset stream to start
    file.seek(0)

    # Get MIME type from file header
    mime = magic.Magic(mime=True)
    mime_type = mime.from_buffer(file.read(64))
    file.seek(0)

    # Map MIME type to friendly label
    mime_map = {
        "application/pdf": "pdf",
        "text/plain": "txt",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    }

    return mime_map.get(mime_type, "unknown")


class VectorDB:
    def __init__(self, collection_name: str) -> None:
        client = chromadb.PersistentClient(path="chroma")
        self.vector_store = client.get_or_create_collection(collection_name)

    def add_file(self, file: BytesIO, file_type: str | None = None):
        if file_type is None:
            file_type = detect_file_type(file)
        if file is not None and file_type is not None:
            if file_type == "application/pdf":
                print("Pdf!")
                data = read_pdf(file)
            elif file_type == "text/plain":
                print("Txt!")
                data = file.read().decode(errors="ignore")
            elif (
                file_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                print("Docx!")
                doc = docx.Document(file)
                data = "\n".join([p.text for p in doc.paragraphs])
            elif (
                file_type
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ):
                print("Xlsx!")
            else:
                print("Unknown type")
                exit()
            # chunks = chunk_text(data)
            # [
            #     self.vector_store.upsert(
            #         documents=chunk, ids=sha3_256(chunk.encode("utf-8")).hexdigest()
            #     )
            #     for chunk in chunks.copy()
            # ]
            self.vector_store.upsert(
                documents=data, ids=sha3_256(data.encode("utf-8")).hexdigest()
            )
            print("Ready!")

    def delete_all_data(self, user_id) -> None:
        self.vector_store.delete(where={"user_id": user_id})
