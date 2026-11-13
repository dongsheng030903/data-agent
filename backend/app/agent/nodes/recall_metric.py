from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.entities.metric_info import MetricInfo
from app.prompt.prompt_loader import load_prompt
from app.utils.llm_json import ainvoke_json


async def recall_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "召回指标", "status": "running"})

    query = state["query"]
    keywords = state["keywords"]

    embedding_client = runtime.context["embedding_client"]
    metric_qdrant_repository = runtime.context["metric_qdrant_repository"]

    try:
        # 使用 LLM 扩展关键词；解析失败时降级为原始关键词
        try:
            prompt_template = PromptTemplate(
                template=load_prompt("extend_keywords_for_metric_recall"),
                input_variables=["query"],
            )
            prompt_text = prompt_template.format(query=query)
            expanded = await ainvoke_json(
                llm, prompt_text, timeout=30, retries=1, node_name="recall_metric"
            )
            if not isinstance(expanded, list):
                raise ValueError(f"期望 JSON 数组，实际为 {type(expanded).__name__}")
            expanded = [k for k in expanded if isinstance(k, str) and k.strip()]
        except Exception as e:
            logger.warning(
                f"召回指标关键词扩展失败，降级为原始关键词: {type(e).__name__}: {e}"
            )
            expanded = []

        # 使用扩展后的关键词召回指标信息
        retrieved_metrics_map: dict[str, MetricInfo] = {}
        keywords = list(set(keywords + expanded))
        logger.info(f"召回指标信息扩展关键词：{keywords}")
        for keyword in keywords:
            embedding = await embedding_client.aembed_query(keyword)
            payloads: list[MetricInfo] = await metric_qdrant_repository.search(embedding)
            for payload in payloads:
                metric_id = payload.id
                if metric_id not in retrieved_metrics_map:
                    retrieved_metrics_map[metric_id] = payload

        retrieved_metrics = list(retrieved_metrics_map.values())

        writer({"type": "progress", "step": "召回指标", "status": "success"})
        logger.info(f"召回指标信息：{list(retrieved_metrics_map.keys())}")
        return {"retrieved_metrics": retrieved_metrics}
    except Exception as e:
        writer({"type": "progress", "step": "召回指标", "status": "error"})
        logger.error(f"召回指标信息失败: {str(e)}")
        raise
