import yaml
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt
from app.utils.llm_json import ainvoke_json


async def filter_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "过滤指标", "status": "running"})

    query = state["query"]
    metric_infos = state["metric_infos"]

    # 候选指标很少时，过滤收益低、但会多一次 LLM 调用；直接保留全部候选
    if len(metric_infos) <= 3:
        logger.info(f"候选指标仅 {len(metric_infos)} 个，跳过指标过滤")
        writer({"type": "progress", "step": "过滤指标", "status": "success"})
        return {"metric_infos": metric_infos}

    try:
        prompt_template = PromptTemplate(
            template=load_prompt("filter_metric_info"),
            input_variables=["query", "metric_infos"],
        )
        prompt_text = prompt_template.format(
            query=query,
            metric_infos=yaml.dump(metric_infos, allow_unicode=True, sort_keys=False),
        )
        result = await ainvoke_json(
            llm,
            prompt_text,
            timeout=30,
            retries=1,
            node_name="filter_metric",
        )
        if not isinstance(result, list):
            raise ValueError(f"期望 JSON 数组，实际为 {type(result).__name__}")
        result = [name for name in result if isinstance(name, str)]

        # 利用模型输出过滤 metric_infos
        for metric_info in metric_infos[:]:
            if metric_info["name"] not in result:
                metric_infos.remove(metric_info)

        writer({"type": "progress", "step": "过滤指标", "status": "success"})
        logger.info(f"过滤后的指标: {[metric_info['name'] for metric_info in metric_infos]}")
        return {"metric_infos": metric_infos}
    except Exception as e:
        # 降级：保留全部候选指标，保证主链路继续执行
        logger.warning(f"过滤指标失败，降级为保留全部候选指标: {type(e).__name__}: {e}")
        writer({"type": "progress", "step": "过滤指标", "status": "success"})
        return {"metric_infos": metric_infos}
