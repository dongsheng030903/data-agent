from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.entities.value_info import ValueInfo
from app.prompt.prompt_loader import load_prompt
from app.utils.llm_json import ainvoke_json


async def recall_value(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "召回字段取值", "status": "running"})

    query = state["query"]
    keywords = state["keywords"]

    value_es_repository = runtime.context["value_es_repository"]

    try:
        # 使用 LLM 扩展关键词；解析失败时降级为原始关键词
        try:
            prompt_template = PromptTemplate(
                template=load_prompt("extend_keywords_for_value_recall"),
                input_variables=["query"],
            )
            prompt_text = prompt_template.format(query=query)
            expanded = await ainvoke_json(
                llm, prompt_text, timeout=30, retries=1, node_name="recall_value"
            )
            if not isinstance(expanded, list):
                raise ValueError(f"期望 JSON 数组，实际为 {type(expanded).__name__}")
            expanded = [k for k in expanded if isinstance(k, str) and k.strip()]
        except Exception as e:
            logger.warning(
                f"召回字段取值关键词扩展失败，降级为原始关键词: {type(e).__name__}: {e}"
            )
            expanded = []

        # 使用扩展后的关键词召回字段取值
        values_map: dict[str, ValueInfo] = {}
        keywords = list(set(keywords + expanded))
        logger.info(f"召回字段取值扩展关键词：{keywords}")
        for keyword in keywords:
            values: list[ValueInfo] = await value_es_repository.search(keyword)
            for value in values:
                value_id = value.id
                if value_id not in values_map:
                    values_map[value_id] = value

        retrieved_values = list(values_map.values())

        writer({"type": "progress", "step": "召回字段取值", "status": "success"})
        logger.info(f"召回字段取值：{list(values_map.keys())}")

        return {"retrieved_values": retrieved_values}
    except Exception as e:
        writer({"type": "progress", "step": "召回字段取值", "status": "error"})
        logger.error(f"召回字段取值失败: {str(e)}")
        raise
