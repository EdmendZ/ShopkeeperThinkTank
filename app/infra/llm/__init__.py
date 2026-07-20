"""模型基础设施包，只向上层公开统一的 ``llm_provider`` 入口。"""
from app.infra.llm.providers import llm_provider

# __all__ 明确 ``from ... import *`` 时允许导出的公开名称，也可作为包的公开 API 清单。
__all__ = ["llm_provider"]
