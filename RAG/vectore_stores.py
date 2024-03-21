from pathlib import Path

import chromadb
from llama_index.legacy.vector_stores import ChromaVectorStore


class ChromaVS:
    def __init__(self):
        self._db = chromadb.PersistentClient(path=str(Path(__file__).parent / 'gospodderzka_db_e5'))
        self._chroma_collection = self._db.get_collection(name="main")
        self._llama_chroma_wrapper = ChromaVectorStore(chroma_collection=self._chroma_collection)

    @property
    def chroma_vector_store(self):
        return self._llama_chroma_wrapper

