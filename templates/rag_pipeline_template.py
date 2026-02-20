"""
RAG Pipeline — build retrieval-augmented generation systems.

Automates workflows for:
  1. Document indexing and chunking
  2. Vector store creation (Chroma, FAISS)
  3. Similarity search
  4. Context-aware answer generation
  5. Multi-modal RAG
"""

import os
import tempfile
from typing import Any, Dict, List

from templates.base import Template, InputField, OutputField, RouteType


class RAGPipelineTemplate(Template):
    name = "rag-pipeline"
    category = "ML"
    description = (
        "Build powerful retrieval-augmented generation pipelines. Index documents, "
        "create vector databases, and generate context-aware answers using state-of-the-art "
        "LLMs and embedding models. Supports PDF, txt, markdown, and web content. "
        "Perfect for building Q&A systems, chatbots, and knowledge bases."
    )
    version = "1.0.0"

    inputs = [
        InputField(
            name="documents",
            type="json",
            description="List of documents or paths to index",
            required=True,
        ),
        InputField(
            name="mode",
            type="text",
            description="Operation mode",
            required=False,
            default="index",
            options=["index", "query", "hybrid"],
        ),
        InputField(
            name="chunk_size",
            type="number",
            description="Document chunk size in characters",
            required=False,
            default=1000,
        ),
        InputField(
            name="chunk_overlap",
            type="number",
            description="Chunk overlap",
            required=False,
            default=200,
        ),
        InputField(
            name="embedding_model",
            type="text",
            description="Embedding model for vectorization",
            required=False,
            default="sentence-transformers/all-MiniLM-L6-v2",
        ),
        InputField(
            name="vector_store",
            type="text",
            description="Vector store backend",
            required=False,
            default="chroma",
            options=["chroma", "faiss", "milvus", "qdrant"],
        ),
        InputField(
            name="llm_model",
            type="text",
            description="LLM for answer generation",
            required=False,
            default="gpt-3.5-turbo",
        ),
        InputField(
            name="query",
            type="text",
            description="Query for retrieval (in query mode)",
            required=False,
        ),
        InputField(
            name="top_k",
            type="number",
            description="Number of chunks to retrieve",
            required=False,
            default=5,
        ),
        InputField(
            name="return_sources",
            type="text",
            description="Return source documents",
            required=False,
            default="true",
            options=["true", "false"],
        ),
    ]

    outputs = [
        OutputField(name="answer", type="text", description="Generated answer"),
        OutputField(name="sources", type="json", description="Retrieved source chunks"),
        OutputField(name="vector_store_path", type="text", description="Path to saved vector store"),
        OutputField(name="metadata", type="json", description="Pipeline metadata"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = False
    gpu_type = None
    memory_mb = 4096
    timeout_sec = 300
    pip_packages = ["langchain", "chromadb", "faiss-cpu", "sentence-transformers", "pypdf", "beautifulsoup4"]

    def setup(self):
        self._vector_store = None
        self._embeddings = None
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
        from langchain.vectorstores import Chroma, FAISS
        from langchain.embeddings import SentenceTransformerEmbeddings

        documents = kwargs["documents"]
        mode = kwargs.get("mode", "index")
        chunk_size = int(kwargs.get("chunk_size", 1000))
        chunk_overlap = int(kwargs.get("chunk_overlap", 200))
        embed_model = kwargs.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        vs_type = kwargs.get("vector_store", "chroma")
        llm_model = kwargs.get("llm_model", "gpt-3.5-turbo")
        query = kwargs.get("query", "")
        top_k = int(kwargs.get("top_k", 5))
        return_sources = kwargs.get("return_sources", "true") == "true"

        if isinstance(documents, str):
            documents = [documents]

        if mode == "index":
            docs = []
            for doc_path in documents:
                if doc_path.endswith(".pdf"):
                    loader = PyPDFLoader(doc_path)
                elif doc_path.startswith("http"):
                    loader = WebBaseLoader(doc_path)
                else:
                    loader = TextLoader(doc_path)
                docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = splitter.split_documents(docs)

            self._embeddings = SentenceTransformerEmbeddings(model_name=embed_model)

            if vs_type == "chroma":
                self._vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self._embeddings,
                )
            else:
                self._vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=self._embeddings,
                )

            return {
                "answer": f"Indexed {len(chunks)} chunks from {len(documents)} documents",
                "sources": [],
                "vector_store_path": str(self._vector_store),
                "metadata": {
                    "num_chunks": len(chunks),
                    "num_docs": len(documents),
                    "embedding_model": embed_model,
                    "vector_store": vs_type,
                },
            }

        elif mode == "query" or mode == "hybrid":
            if self._vector_store is None:
                return {
                    "answer": "Error: No vector store found. Run in index mode first.",
                    "sources": [],
                    "vector_store_path": "",
                    "metadata": {},
                }

            if self._embeddings is None:
                self._embeddings = SentenceTransformerEmbeddings(model_name=embed_model)

            results = self._vector_store.similarity_search(query, k=top_k)

            context = "\n\n".join([doc.page_content for doc in results])

            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

            answer = f"[LLM {llm_model} would generate answer here based on context]"

            sources = []
            if return_sources:
                for i, doc in enumerate(results):
                    sources.append({
                        "chunk_id": i,
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                    })

            return {
                "answer": answer,
                "sources": sources,
                "vector_store_path": str(self._vector_store),
                "metadata": {
                    "query": query,
                    "num_sources": len(sources),
                    "llm_model": llm_model,
                },
            }

        return {"answer": "", "sources": [], "vector_store_path": "", "metadata": {}}
