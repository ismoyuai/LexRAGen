# -*- coding: utf-8 -*-
import chromadb
import time
from core.config import Config
from llama_index.core import VectorStoreIndex, StorageContext ,Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from typing import List, Optional
from llama_index.core.schema import TextNode


class VectorStoreManager:
    """向量存储管理类"""

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def init_index(self, nodes: Optional[List[TextNode]] = None) -> VectorStoreIndex:
        """初始化或加载向量索引"""
        start_time = time.time()
        storage_context = self._create_storage_context()

        if self.collection.count() == 0 and nodes:
            print(f"创建新索引（{len(nodes)}节点）...")
            self._build_new_index(nodes, storage_context)
        else:
            print("加载已有索引...")
            storage_context = StorageContext.from_defaults(
                persist_dir=Config.PERSIST_DIR,
                vector_store=ChromaVectorStore(chroma_collection=self.collection)
            )

        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

        self._validate_storage(storage_context)
        print(f"索引加载耗时：{time.time() - start_time:.2f}s")
        return index

    def _create_storage_context(self):
        """创建存储上下文"""
        return StorageContext.from_defaults(
            vector_store=ChromaVectorStore(chroma_collection=self.collection)
        )

    def _build_new_index(self, nodes: List[TextNode], storage_context: StorageContext):
        """构建新索引"""
        storage_context.docstore.add_documents(nodes)
        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        storage_context.persist(persist_dir=Config.PERSIST_DIR)

    def _validate_storage(self, storage_context: StorageContext):
        """验证存储状态"""
        doc_count = len(storage_context.docstore.docs)
        print("\n存储验证：")
        print(f"文档数量：{doc_count}")
        if doc_count > 0:
            sample_key = next(iter(storage_context.docstore.docs.keys()))
            print(f"示例节点ID：{sample_key}")