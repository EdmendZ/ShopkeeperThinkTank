# Python 代码注解全量更新设计

## 目标

在不改变运行行为的前提下，全量检查项目1与项目2的 Python 源码，使注解准确反映当前实现，并形成可在后续迭代中复用的注解规则。

## 范围

- `app/**/*.py`
- `tests/**/*.py`
- `test/**/*.py` 中实际受版本控制的 Python 文件
- `main.py`
- 包含 package `__init__.py`、HTTP API、LangGraph 工作流、RAG service、基础设施适配器、共享配置与工具、测试和演示入口

不修改 `.env`、Prompt、HTML、依赖锁文件以及生成目录。除非注解验证暴露出语法错误，否则不修改业务逻辑、公开接口签名、配置值或测试断言。

## 注解策略

采用“接口优先 + 关键逻辑补充”：

1. 非空模块使用 module Docstring 说明职责、边界和主要 side effects。
2. public class/function/method 使用 Docstring 说明业务目的，并按实际需要补充 `Args`、`Returns`、`Raises` 和 `Side Effects`。
3. 行内注释仅解释代码本身不能直接表达的信息，包括：
   - LangGraph state 的字段流转与 partial state 返回语义；
   - RAG、HyDE、RRF、Embedding、Reranker 等算法选择意图；
   - Milvus、MinIO、MongoDB、MinerU、FastAPI 等外部系统约束；
   - 幂等性、缓存、批处理、超时、降级和异常边界；
   - 为何必须采用当前顺序、默认值或兼容分支。
4. 删除或改写逐行复述赋值、循环、函数调用的注释，以及与当前返回结构或字段名称不一致的旧注释。
5. 中文描述业务含义，API、framework、protocol、algorithm、data type 等标准术语保留英文；同一术语在两个仓库中保持一致。

## 双仓库同步

两个项目当前大部分 Python 文件同源。对实现完全相同的文件应用相同注解；对已有实现差异的文件分别核对，不用一方内容覆盖另一方。

所有编辑必须保留两个工作区现有未提交修改。修改前后都检查 `git diff`，只在目标行附近做最小化 patch，禁止 reset、checkout 或整文件覆盖。

## 自我学习闭环

结合 `self-improving-agent`，按 capture-first 原则执行：

1. Working memory 记录本轮范围、风格、约束和当前发现。
2. 完成后写入 episodic memory，记录实际遇到的过时注解类型、有效处理方式、验证结果和用户反馈。
3. 将可复用规则作为低风险候选写入 semantic memory，并记录来源、置信度和应用次数。
4. 单次经验不直接改写全局 `SKILL.md`、`AGENTS.md` 或运行时代码；只有重复出现、通过验证或用户明确授权后才推广。
5. 使用 self-validation checklist 检查示例、仓库约定、重复规则和冲突指导。

## 验证

- 对全部目标 Python 文件运行 `compileall`，确认注解编辑没有引入语法或缩进错误。
- 分别运行两个项目的现有测试；依赖外部服务而无法执行的测试需明确记录，不伪造成功结果。
- 使用 AST 检查非空模块和 public API 的 Docstring 覆盖情况，并人工复核缺失项是否属于刻意保留的简单 package marker。
- 搜索 `TODO`、`FIXME`、`HACK`、旧字段名和明显教学占位语句，逐项确认保留理由。
- 比较两个项目的共有文件：实现相同的文件应保持注解一致，实现不同的文件应保留差异。
- 最终 `git diff` 只应包含注解、Docstring、必要的空行/文件尾换行，以及用户原有未提交业务改动。

## 完成标准

- 全部目标 Python 文件均经过人工或结构化审查。
- 注解与当前控制流、输入输出、异常和 side effects 一致。
- 关键模块可通过 Docstring 理解职责，无需依赖逐行复述注释。
- 两个项目的共有实现不产生新的注解漂移。
- 编译检查通过；测试结果与外部依赖限制均有证据。
- 自我学习记录可追溯，但未未经验证推广为全局硬规则。
