import yaml
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt
from app.utils.llm_json import ainvoke_json


async def filter_table(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "过滤表格", "status": "running"})

    query = state["query"]
    table_infos = state["table_infos"]

    # 候选表/字段规模很小时，过滤收益低、但会多一次 LLM 调用；直接保留全部候选
    total_columns = sum(len(table.get("columns", [])) for table in table_infos)
    if len(table_infos) <= 5 and total_columns <= 40:
        logger.info(
            f"候选表 {len(table_infos)} 张、字段 {total_columns} 个，跳过表格过滤"
        )
        writer({"type": "progress", "step": "过滤表格", "status": "success"})
        return {"table_infos": table_infos}

    try:
        prompt_template = PromptTemplate(
            template=load_prompt("filter_table_info"),
            input_variables=["query", "table_infos"],
        )
        prompt_text = prompt_template.format(
            query=query,
            table_infos=yaml.dump(table_infos, allow_unicode=True, sort_keys=False),
        )
        result = await ainvoke_json(
            llm,
            prompt_text,
            timeout=30,
            retries=1,
            node_name="filter_table",
        )
        if not isinstance(result, dict) or not result:
            raise ValueError(f"期望非空 JSON 对象，实际为 {type(result).__name__}")

        # 利用模型输出过滤 table_infos
        for table_info in table_infos[:]:
            if table_info["name"] not in result:
                table_infos.remove(table_info)
            else:
                selected_columns = result[table_info["name"]]
                if not isinstance(selected_columns, list):
                    raise ValueError(f"表 {table_info['name']} 的字段列表不是数组")
                for column_info in table_info["columns"][:]:
                    if column_info["name"] not in selected_columns:
                        table_info["columns"].remove(column_info)

        if not table_infos:
            raise ValueError("过滤后没有剩余表")

        writer({"type": "progress", "step": "过滤表格", "status": "success"})
        logger.info(f"过滤后的表信息: {[table_info['name'] for table_info in table_infos]}")
        return {"table_infos": table_infos}
    except Exception as e:
        # 降级：保留全部候选表，保证主链路继续执行
        logger.warning(f"过滤表失败，降级为保留全部候选表: {type(e).__name__}: {e}")
        writer({"type": "progress", "step": "过滤表格", "status": "success"})
        return {"table_infos": table_infos}
