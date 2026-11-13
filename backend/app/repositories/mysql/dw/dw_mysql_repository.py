import asyncio
import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class DWMySQLRepository:
    FORBIDDEN_SQL = re.compile(
        r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|replace|merge|call|load_file|outfile)\b",
        re.IGNORECASE,
    )

    def __init__(self, session: AsyncSession):
        self.session = session

    def _ensure_read_only(self, sql: str) -> str:
        stripped = sql.strip().rstrip(";").strip()
        if not re.match(r"^(select|with)\b", stripped, re.IGNORECASE):
            raise ValueError("只允许执行 SELECT 查询")
        if ";" in stripped:
            raise ValueError("不允许执行多条 SQL")
        if self.FORBIDDEN_SQL.search(stripped):
            raise ValueError("检测到非只读 SQL 关键字")
        return stripped

    async def get_column_types(self, table_name: str) -> dict[str, str]:
        sql = f"show columns from {table_name}"
        result = await self.session.execute(text(sql))
        return {row.Field: row.Type for row in result.fetchall()}

    async def get_column_values(self, table_name: str, column_name: str, limit: int):
        sql = f"select distinct {column_name} from {table_name} limit {limit}"
        result = await self.session.execute(text(sql))
        return result.scalars().fetchall()

    async def get_db_info(self):
        result = await self.session.execute(text("select version()"))
        version = result.scalar()

        dialect = self.session.get_bind().dialect.name

        return {'version': version, 'dialect': dialect}

    async def validate_sql(self, sql):
        sql = self._ensure_read_only(sql)
        await asyncio.wait_for(self.session.execute(text(f"explain {sql}")), timeout=30)

    async def execute_sql(self, sql):
        sql = self._ensure_read_only(sql)
        result = await asyncio.wait_for(self.session.execute(text(sql)), timeout=30)
        return [dict(row) for row in result.mappings().fetchmany(1000)]
