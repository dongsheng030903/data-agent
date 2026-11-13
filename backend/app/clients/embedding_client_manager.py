from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.conf.app_config import EmbeddingConfig, app_config


class EmbeddingClientManager:
    def __init__(self, config: EmbeddingConfig):
        self.client: Optional[Embeddings] = None
        self.config = config

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def init(self):
        if self.config.provider == "ollama":
            # Ollama 提供 OpenAI 兼容的 /v1/embeddings 接口
            self.client = OpenAIEmbeddings(
                model=self.config.model,
                base_url=f"{self._get_url()}/v1",
                api_key="ollama",
                check_embedding_ctx_length=False,
            )
        else:
            self.client = HuggingFaceEndpointEmbeddings(model=self._get_url())


embedding_client_manager = EmbeddingClientManager(app_config.embedding)
