from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.entities.column_info import ColumnInfo
from app.prompt.prompt_loader import load_prompt
from app.utils.llm_json import ainvoke_json


async def recall_column(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "召回字段", "status": "running"})

    query = state["query"]
    keywords = state["keywords"]

    embedding_client = runtime.context["embedding_client"]
    column_qdrant_repository = runtime.context["column_qdrant_repository"]

    try:
        # 使用 LLM 扩展关键词；解析失败时降级为原始关键词
        try:
            prompt_template = PromptTemplate(
                template=load_prompt("extend_keywords_for_column_recall"),
                input_variables=["query"],
            )
            prompt_text = prompt_template.format(query=query)
            expanded = await ainvoke_json(
                llm, prompt_text, timeout=30, retries=1, node_name="recall_column"
            )
            if not isinstance(expanded, list):
                raise ValueError(f"期望 JSON 数组，实际为 {type(expanded).__name__}")
            expanded = [k for k in expanded if isinstance(k, str) and k.strip()]
        except Exception as e:
            logger.warning(
                f"召回字段关键词扩展失败，降级为原始关键词: {type(e).__name__}: {e}"
            )
            expanded = []

        # 使用扩展后的关键词召回字段信息
        retrieved_columns_map: dict[str, ColumnInfo] = {}
        keywords = list(set(keywords + expanded))
        logger.info(f"召回字段信息扩展关键词：{keywords}")
        for keyword in keywords:
            embedding = await embedding_client.aembed_query(keyword)
            payloads: list[ColumnInfo] = await column_qdrant_repository.search(
                embedding
            )
            for payload in payloads:
                column_id = payload.id
                if column_id not in retrieved_columns_map:
                    retrieved_columns_map[column_id] = payload

        retrieved_columns = list(retrieved_columns_map.values())

        writer({"type": "progress", "step": "召回字段", "status": "success"})
        logger.info(f"召回字段信息：{list(retrieved_columns_map.keys())}")
        return {"retrieved_columns": retrieved_columns}
    except Exception as e:
        writer({"type": "progress", "step": "召回字段", "status": "error"})
        logger.error(f"召回字段信息失败: {str(e)}")
        raise
