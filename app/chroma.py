import chromadb
from io import BytesIO
from hashlib import sha3_256
from tempfile import SpooledTemporaryFile
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHUNK_SIZE = 512
OVERLAP_SIZE = 32


class UploadedFile:
    def __init__(self, file: SpooledTemporaryFile, filename: str):
        self.file = file
        self.filename = filename

    def extract_text(self) -> str:
        buf = BytesIO(self.file.read())
        stream = DocumentStream(name=self.filename, stream=buf)
        converter = DocumentConverter()
        result = converter.convert(stream)
        return result.document.export_to_text()


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP_SIZE) -> list[str]:
    chunks = []
    # Simple character-based chunking
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size]
        if chunk:  # Ensure we don't add empty chunks
            chunks.append(chunk)

    return chunks


class VectorDB:
    def __init__(self, collection_name: str) -> None:
        client = chromadb.PersistentClient(path="chroma")
        self.vector_store = client.get_or_create_collection(collection_name)
        self.collection = Chroma(collection_name, persist_directory="chroma")

    def get_retriever(self):
        return self.collection.as_retriever()

    def add_file(
        self,
        filename: str,
        file: SpooledTemporaryFile,
        file_type: str | None = None,
    ):
        if file is not None:
            if file_type == "text/plain":
                data = file
            else:
                data = UploadedFile(file, filename).extract_text()

            chunks = chunk_text(data)
            # [
            #     self.vector_store.upsert(
            #         documents=chunk, ids=sha3_256(chunk.encode("utf-8")).hexdigest()
            #     )
            #     for chunk in chunks.copy()
            # ]
            # print("Ready!")
            return chunks[0]

    def delete_all_data(self, user_id) -> None:
        self.vector_store.delete(where={"user_id": user_id})

    def add_text(self, text: str) -> None:
        chunks = chunk_text(text)
        [
            self.vector_store.upsert(
                documents=chunk, ids=sha3_256(chunk.encode("utf-8")).hexdigest()
            )
            for chunk in chunks.copy()
        ]
        print("Ready!")

    def query(self, input: str) -> list[Document]:
        return self.collection.similarity_search(input, k=3)


if __name__ == "__main__":
    vectordb = VectorDB("docling_file")
