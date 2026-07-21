"""
Markdown 文档切分服务。

这个模块接收图片处理完成后的 Markdown，把一篇长文整理成多个适合检索的 Chunk：
1. 从 LangGraph 的共享 state 中取得正文和文件标题；
2. 先按 Markdown 标题切成章节；
3. 再拆分过长章节，并尝试合并过短片段；
4. 把结果备份为 chunks.json；
5. 将 Chunk 列表写回 state，交给后续的主体识别、向量化和 Milvus 入库节点。

这里的长度都使用 Python 的 len() 按“字符数”计算，不等同于大模型的 Token 数。
"""

import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.shared.runtime.logger import logger, step_log
from app.rag.import_.config import CHUNK_MAX_SIZE, CHUNK_SIZE
from app.process.import_.agent.state import ImportGraphState


# -------------------------- Service 总入口：串联完整切分流程 --------------------------
# @step_log 是项目自定义装饰器：调用函数前后会自动打印开始、结束和耗时日志；
# 如果函数抛出异常，装饰器会记录异常后继续向上抛出，不会把错误悄悄吞掉。
@step_log("split_document")
def split_document(state: dict) -> dict:
    """
    文档切分的总入口，负责按固定顺序调用各个子函数。

    state 是 LangGraph 节点之间共同传递的普通字典。本函数主要读取：
    - md_content：图片处理完成后的 Markdown 正文；
    - md_path：Markdown 文件路径，也是 chunks.json 的保存位置依据；
    - file_title：原始文档标题。

    处理完成后会新增或更新 state["chunks"]，其结构大致为：
    [{"title": "...", "content": "...", "parent_title": "...", "part": 0, ...}]

    Args:
        state: 导入流程共享状态，至少要能取得正文，并提供 md_path 用于备份。

    Returns:
        原来的 state 字典对象，只是其中已经写入最终 Chunk 列表。
    """
    # 第一步：准备切分所需的正文和标题。
    # tuple 可以一次返回多个值；左侧的两个变量会按位置接收这两个返回值。
    md_content, file_title = load_markdown_content(state)

    # 第二步：按 #、##、### 等一至六级标题做“粗切分”。
    # 这一步优先保留章节边界，还没有处理过长或过短的问题。
    chunks = split_by_titles(md_content, file_title)

    # 第三步：继续整理 Chunk 长度。
    # CHUNK_MAX_SIZE 是目标最大字符数；CHUNK_SIZE 只是触发短块合并的阈值，
    # 不是硬性下限，所以最后一个 Chunk 仍有可能小于 CHUNK_SIZE。
    chunks = refine_chunks(chunks, max_len=CHUNK_MAX_SIZE, min_len=CHUNK_SIZE)

    # 第四步：把结果保存到 Markdown 同目录的 chunks.json，方便人工检查。
    # 备份发生在 state 回写之前；如果文件写入失败，异常会直接中断当前流程。
    backup_chunks(chunks, state["md_path"])

    # 第五步：把最终结果放回共享状态，供下游节点继续使用。
    state["chunks"] = chunks

    return state


# -------------------------- 子函数 1：读取正文并补齐标题 --------------------------
@step_log("load_markdown_content")
def load_markdown_content(state: dict) -> tuple[str, str]:
    """
    从 state 中取得 Markdown 正文和文档标题，并统一换行符。

    读取优先级：
    1. 优先使用 state["md_content"]，避免重复读取磁盘；
    2. 正文为空时，根据 state["md_path"] 读取 UTF-8 文件；
    3. 标题为空时，使用 Markdown 文件名（不含 .md 后缀）；
    4. Windows 的 \r\n 和旧式 \r 换行统一转成 \n。

    Args:
        state: 导入流程共享状态。

    Returns:
        tuple[str, str] 表示一个固定包含两项的元组：
        第一项是处理后的 Markdown 正文，第二项是文档标题。

    Raises:
        UnicodeError: 文件内容不是有效的 UTF-8 文本。
        ValueError: state 没有正文，且 md_path 为空、不是文件或文件内容为空。
    """
    # dict.get("键", 默认值) 在键不存在时返回默认值，不会像 state["键"] 那样
    # 立即抛出 KeyError，适合读取可能缺失的流程字段。
    md_content = state.get("md_content", "")
    file_title = state.get("file_title", "")
    md_path = state.get("md_path", "")

    # Python 会把空字符串判断为 False，因此 not md_content 表示“正文为空”。
    if not md_content:
        logger.warning("没有从state读取到md_content内容,我们使用md_path尝试再次读取!")

        # 只有在 state 没有直接提供正文时，md_path 才是必须参数。
        # is_file() 会同时确认路径存在且确实是普通文件，目录也会被判定为无效。
        md_path_obj = Path(md_path) if md_path else None
        if md_path_obj is None or not md_path_obj.is_file():
            logger.error("md_content为空，且md_path为空或没有对应文件，业务无法继续!")
            raise ValueError("md_content为空，且md_path为空或没有对应文件，业务无法继续!")

        # 路径校验通过后再读取，避免把底层 FileNotFoundError 暴露为不明确的流程错误。
        md_content = md_path_obj.read_text(encoding="utf-8")
        # 回填 state 后，后续节点可以直接使用正文，不必再次访问磁盘。
        state["md_content"] = md_content

        # 空文件虽然路径有效，但仍然没有可切分内容。
        if not md_content:
            raise ValueError("md_content没数据,并且尝试读取md_path依然没有数据,终止执行!!")

    # Path.stem 只保留文件名主体，例如 "manual.md" 会得到 "manual"。
    # 条件表达式 A if 条件 else B 表示：条件成立取 A，否则取 B。
    if not file_title:
        file_title = Path(md_path).stem if md_path else "default"
        state["file_title"] = file_title

    # 必须先替换 \r\n，再处理单独的 \r；反过来会把一个 Windows 换行变成两个 \n。
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

    return md_content, file_title


# -------------------------- 子函数 2：按 Markdown 标题粗切分 --------------------------
@step_log("split_by_titles")
def split_by_titles(md_content: str, file_title: str) -> list[dict]:
    """
    从上到下扫描 Markdown，遇到标题时结束上一块并开始下一块。

    例如输入：
        # 产品介绍
        正文 A
        ## 安全说明
        正文 B

    会得到两个字典组成的列表，每个字典保存：
    - content：标题和正文拼成的完整 Chunk；
    - title：当前 Markdown 标题；
    - file_title：该 Chunk 来自哪一份文档。

    Args:
        md_content: 已经统一换行符的 Markdown 正文。
        file_title: 文档级标题，会复制到每一个 Chunk 中用于溯源。

    Returns:
        list[dict] 表示“列表中的每个元素都是字典”。如果全文没有 Markdown
        标题，就返回一个 title 为 "default"、content 为全文的 Chunk。

    注意：
        代码围栏内部的 # 不会当成标题。首个标题前的非空内容会保存为独立的
        “文档前言” Chunk；中间若只有标题而没有正文，仍不会单独生成 Chunk。
    """
    # re.compile 会提前编译正则，后面的每一行都可以重复使用它。
    # 正则逐段解释：
    # ^            必须从行首开始匹配
    # \s*          允许标题前存在空白
    # #{1,6}       匹配 1 到 6 个 #，即 Markdown 的一至六级标题
    # \s.+         # 后必须有空格和实际标题内容，因此 "#标题" 不会匹配
    reg = re.compile(r"^\s*#{1,6}\s.+")

    # split("\n") 把整篇字符串拆成行列表，便于按顺序维护“当前章节”。
    lines = md_content.split("\n")

    # list[dict] 是类型注解，只帮助读者、编辑器和检查工具理解数据结构，
    # 运行时它仍然是一个普通的 Python 列表。
    chunks: list[dict] = []

    # 下面三个变量共同组成一个简单的“状态机”：
    # current_title 记录当前章节标题；None 表示还没遇到第一个标题。
    current_title = None
    # current_title_lines 暂存当前章节已经扫描到的所有行。
    current_title_lines: list[str] = []
    # is_code_block 记录当前是否位于 ``` 或 ~~~ 围起来的代码块中。
    is_code_block = False
    # chunk_size 实际记录“遇到过多少个标题”，用于判断全文是否完全没有标题。
    chunk_size = 0

    # for 循环会按原顺序逐行处理 Markdown。
    for raw_line in lines:
        # strip() 去掉行首、行尾空白。这样更容易识别标题，但也意味着最终 Chunk
        # 不会逐字保留原文的行首缩进和行尾空格。
        line = raw_line.strip()

        # ----------------------- 1. 识别代码围栏 -----------------------
        # 每遇到一次开始/结束围栏，就用 not 把布尔值 True、False 相互切换。
        if line.startswith("```") or line.startswith("~~~"):
            is_code_block = not is_code_block
            # 围栏行本身仍属于正文，所以要保存到当前章节。
            current_title_lines.append(line)
            # continue 跳过本轮剩余代码，避免把围栏行继续当作标题判断。
            continue

        # ----------------------- 2. 遇到新标题 -----------------------
        # reg.match(line) 返回匹配对象或 None；and 要求当前还不能处于代码块中。
        if reg.match(line) and not is_code_block:
            # 第一次遇到标题时，current_title 仍为 None；此前累计的行就是文档前言。
            # strip() 只清理前言整体两端的空白，正文中间的空行仍会保留。
            if current_title is None:
                preamble_content = "\n".join(current_title_lines).strip()
                # 标题前只有空行时，strip() 的结果为空，不应生成没有内容的 Chunk。
                if preamble_content:
                    chunks.append({
                        "content": preamble_content,
                        "title": "文档前言",
                        "file_title": file_title
                    })

            # 新标题出现时，先保存已经收集完的上一章节。
            # len > 1 表示除了标题外至少还有一行，只有标题的中间章节会被跳过。
            if current_title and len(current_title_lines) > 1:
                chunks.append({
                    # join 把行列表重新连接成带换行符的完整字符串。
                    "content": "\n".join(current_title_lines),
                    "title": current_title,
                    "file_title": file_title
                })

            # 当前标题成为下一块的第一行，并重新开始收集正文。
            # 首次进入本分支时，前言已经在上方单独结算，不会与标题正文混在一起。
            current_title = line
            current_title_lines = [current_title]
            chunk_size += 1

        # ----------------------- 3. 遇到普通正文 -----------------------
        # 普通行以及代码块中形似标题的行，都继续追加到当前章节。
        else:
            current_title_lines.append(line)

    # ----------------------- 4. 保存最后一个章节 -----------------------
    # 最后一块后面没有“下一个标题”帮助触发保存，所以必须在循环外单独追加。
    if current_title:
        chunks.append({
            "content": "\n".join(current_title_lines),
            "title": current_title,
            "file_title": file_title
        })

    # ----------------------- 5. 全文无标题的兜底处理 -----------------------
    # 没有任何标题时，直接把原始全文作为一个 Chunk，不使用逐行 strip 后的内容。
    if chunk_size == 0:
        chunks.append({
            "content": md_content,
            "title": "default",
            "file_title": file_title
        })

    return chunks


# -------------------------- 子函数 3：统一整理 Chunk 长度和字段 --------------------------
@step_log("refine_chunks")
def refine_chunks(
    sections: list[dict],
    max_len: int = CHUNK_MAX_SIZE,
    min_len: int = CHUNK_SIZE,
) -> list[dict]:
    """
    对标题粗切结果做第二轮整理，使 Chunk 更适合后续检索。

    本函数分为三个阶段：
    1. 超过 max_len 的章节交给 _split_long_section 继续拆分；
    2. 短于 min_len 的当前块交给 _merge_short_sections 尝试向后合并；
    3. 为每个结果补齐 part 和 parent_title 字段。

    Args:
        sections: split_by_titles 返回的标题章节列表。
        max_len: Chunk 的目标最大字符数；小于等于 0 时直接跳过整理。
        min_len: 触发短块合并的字符数，不是最终 Chunk 的强制最小长度。

    Returns:
        整理后的 Chunk 列表。

    注意：
        Python 字典是可变对象。未发生长块拆分时，结果会继续引用原来的字典；
        后面的字段补齐和短块合并可能直接修改这些原字典。
    """
    # ----------------------- 1. 检查最大长度配置 -----------------------
    # not max_len 可以识别 0、None 等“空值”；or 表示任一条件成立就进入分支。
    # 此处直接返回原列表，连 part、parent_title 的补齐步骤也会一起跳过。
    if not max_len or max_len <= 0:
        logger.warning(f"步骤4：Chunk最大长度配置无效（{max_len}），跳过精细化处理")
        return sections

    # refined_split 用于保存“长块拆开后”的中间结果。
    refined_split = []

    # _split_long_section 的返回值始终是列表。
    # extend 会把子列表中的元素逐个加入 refined_split；如果使用 append，
    # 结果会变成 [[块1, 块2], [块3]] 这样的嵌套列表。
    for sec in sections:
        refined_split.extend(_split_long_section(sec, max_len))

    # ----------------------- 2. 尝试合并短块 -----------------------
    # 合并算法只从前向后扫描，因此最后一个短块可能找不到后继块而原样保留。
    final_sections = _merge_short_sections(refined_split, min_length=min_len, max_length=max_len)

    # ----------------------- 3. 补齐下游需要的元数据 -----------------------
    for sec in final_sections:
        # 没有 part 说明该章节没有被二次拆分，用 0 表示“原始完整章节”。
        if "part" not in sec:
            sec["part"] = 0
        # 被二次拆分的块已有 parent_title；普通章节则把自己的 title 当作父标题。
        if not sec.get("parent_title"):
            sec["parent_title"] = sec.get("title") or ""

    return final_sections


# 以下两个以下划线开头的函数是模块内部辅助函数。
# 下划线只是 Python 的命名约定，表示“不建议模块外部直接调用”，并不会禁止调用。
def _split_long_section(section: dict[str, Any], max_length: int = CHUNK_MAX_SIZE) -> list[dict[str, Any]]:
    """
    把一个超过最大长度的章节继续拆成多个子块。

    Args:
        section: 一个章节字典，主要使用 content、title、file_title。
        max_length: 每个子块“标题 + 正文”的目标最大字符数。

    Returns:
        list[dict[str, Any]] 表示“由字典组成的列表”。Any 表示字典值可能是
        字符串、整数等不同类型。章节未超长时也返回列表，只是里面仍是原字典。

    注意：
        如果标题本身已经占满 max_length，正文没有可用长度，本函数会保留原块，
        因此这个特殊结果仍可能超过最大长度。
    """
    # get("content", "") 处理字段缺失，后面的 or "" 还会把 None 转成空字符串。
    content = section.get("content", "") or ""

    # ----------------------- 1. 未超长：直接返回 -----------------------
    # [section] 表示“只包含当前字典的列表”，用于统一函数的返回结构。
    if len(content) <= max_length:
        return [section]

    # ----------------------- 2. 准备标题和正文 -----------------------
    # 即使上游已经处理过换行，这里再次统一，保证该辅助函数单独调用时也能工作。
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    title = section.get("title", "") or ""

    # 条件表达式：有标题时生成“标题 + 两个换行”，没有标题时使用空字符串。
    # 每个子块都会带上这段前缀，所以它也必须占用长度额度。
    prefix = f"{title}\n\n" if title else ""
    available_len = max_length - len(prefix)

    # 标题已经用完整个额度时，RecursiveCharacterTextSplitter 无法接收有效长度。
    if available_len <= 0:
        return [section]

    # 粗切后的 content 通常本来就以标题开头；拆正文前先删掉旧标题，
    # 否则后面添加 prefix 时，每个子块会出现两遍标题。
    body = content
    if title and body.lstrip().startswith(title):
    	body = body[len(title):].lstrip()

    # ----------------------- 3. 创建递归文本切分器 -----------------------
    # 切分器会按 separators 的先后顺序寻找合适边界：
    # 段落 -> 换行 -> 中文句末标点 -> 英文句末标点 -> 空格 -> 最后按字符硬切。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_len,       # 正文额度，不包含后面重新加回的标题
        chunk_overlap=0,                # 相邻子块之间不重复保留上下文
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
    )

    # ----------------------- 4. 组装统一的子块字典 -----------------------
    sub_sections = []
    # enumerate 会同时给出“序号、元素”；start=1 让编号从 1 而不是 0 开始。
    for idx, chunk in enumerate(splitter.split_text(body), start=1):
        text = chunk.strip()
        # 空片段没有检索价值，continue 会跳到下一次循环。
        if not text:
            continue

        # 把标题前缀重新加回正文，strip 再去掉整体两端多余空白。
        full_text = (prefix + text).strip()

        # 这里只复制下游需要的字段；section 中的其他自定义键不会自动保留。
        sub_sections.append({
            "title": f"{title}-{idx}" if title else f"chunk-{idx}",  # 例如“## 安全说明-2”
            "content": full_text,                                    # 标题前缀 + 当前正文
            "parent_title": title,                                   # 拆分前的原章节标题
            "part": idx,                                             # 当前是原章节的第几部分
            "file_title": section.get("file_title"),                 # 来自哪一份文档
        })

    return sub_sections


def _merge_short_sections(
    sections: list[dict[str, Any]],
    min_length: int = CHUNK_SIZE,
    max_length: int = CHUNK_MAX_SIZE,
) -> list[dict[str, Any]]:
    """
    从前向后扫描 Chunk，尝试把当前短块和下一个块合并。

    只有同时满足下面三个条件才会合并：
    1. 当前块长度小于 min_length；
    2. 当前块和下一个块的 parent_title 相同；
    3. 合并后的总长度不超过 max_length。

    Args:
        sections: 已经完成长块拆分的 Chunk 列表。
        min_length: 当前块低于该字符数时，才尝试向后合并。
        max_length: 合并结果允许的最大字符数；小于等于 0 表示不限制。

    Returns:
        保持原顺序的 Chunk 列表。算法每次只比较相邻的两个块，并且只从前向后
        扫描，因此无法保证所有结果都达到 min_length，尤其是列表最后一个短块。

    注意：
        split_by_titles 生成的普通章节此时还没有 parent_title。两个字典调用
        get("parent_title") 都会得到 None，而 None == None 为 True，所以当前
        实现可能把相邻但标题不同的普通章节合并。parent_title 要到本函数返回后
        才由 refine_chunks 补齐。
    """
    # 空列表没有任何内容需要处理，直接返回新空列表。
    if not sections:
        return []

    # merged_sections 保存已经确定不再继续合并的结果。
    merged_sections = []
    # current_chunk 是“等待决定”的当前块，也可以理解为合并过程中的累加器。
    current_chunk = None

    for sec in sections:
        # ----------------------- 1. 初始化累加器 -----------------------
        # 第一个块前面没有可处理内容，先保存到 current_chunk，等待下一个块到来。
        if current_chunk is None:
            current_chunk = sec
            continue

        # ----------------------- 2. 判断能否合并 -----------------------
        current_content = current_chunk.get("content", "")
        # len 返回字符串字符数；严格小于 min_length 才被视为短块。
        is_current_short = len(current_content) < min_length
        # 字段缺失时 get 会返回 None，所以这里也可能是在比较 None == None。
        is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

        if is_current_short and is_same_parent:
            parent_title = sec.get("parent_title", "")
            next_content = sec["content"]

            # 长章节拆出的每个子块都带父标题。合并前删除后一块开头的重复标题，
            # 避免得到“标题 + 正文 A + 标题 + 正文 B”。
            if parent_title and next_content.startswith(parent_title):
                next_content = next_content[len(parent_title):].lstrip()

            # 使用两个换行连接两段内容，让合并后仍保留段落边界。
            merged_content = current_content + "\n\n" + next_content
            # and 具有短路特性：max_length <= 0 时不会再计算后半段长度比较。
            will_exceed_max = max_length > 0 and len(merged_content) > max_length

            # ----------------------- 3A. 合并后过长：拒绝合并 -----------------------
            # 当前块已经确定，加入结果；下一块 sec 成为新的累加器。
            if will_exceed_max:
                merged_sections.append(current_chunk)
                current_chunk = sec
                continue

            # ----------------------- 3B. 条件满足：执行合并 -----------------------
            # 字典是可变对象，这里直接修改 current_chunk 原字典中的 content。
            current_chunk["content"] = merged_content
            # 如果被合并的子块有 part，就记录当前已经合并到第几个子块。
            if "part" in sec:
                current_chunk["part"] = sec["part"]

        else:
            # 当前块不短或父标题不同，不能合并：固定当前块，再处理下一块。
            merged_sections.append(current_chunk)
            current_chunk = sec

    # ----------------------- 4. 循环结束后的收尾 -----------------------
    # current_chunk 一直比循环输出慢一步，最后必须手动追加，否则会漏掉最后一块。
    if current_chunk is not None:
        merged_sections.append(current_chunk)

    return merged_sections


# -------------------------- 子函数 4：备份最终 Chunk --------------------------
@step_log("backup_chunks")
def backup_chunks(chunks: list[dict], md_path: str) -> None:
    """
    把最终 Chunk 列表保存为 Markdown 同目录下的 chunks.json。

    Args:
        chunks: 最终切分结果，里面的内容必须能转换成 JSON。
        md_path: 原 Markdown 文件路径，用来确定备份目录。

    Returns:
        None。这个函数不返回业务数据，而是通过写文件产生结果。

    注意：
        如果 chunks.json 已存在，write_text 会覆盖旧内容；父目录不存在、没有
        写权限或 chunks 中含有无法序列化的对象时，异常会继续向上抛出。
    """
    # Path(md_path).parent 取得 Markdown 所在目录。
    # pathlib 重载了 / 运算符，因此这里的 / 表示拼接路径，不是数学除法。
    chunks_json_path = Path(md_path).parent / "chunks.json"

    # json.dumps 先把 Python 的“列表 + 字典”转换成 JSON 字符串：
    # - ensure_ascii=False：中文保持原样，不转换成 \u4e2d 形式；
    # - indent=4：使用 4 个空格缩进，方便初学者直接打开文件阅读。
    # write_text 再使用 UTF-8 编码把字符串一次性写入磁盘。
    chunks_json_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=4), encoding="utf-8")

    # 下面是等价的传统写法，仅作为学习对照；因为前面有 #，所以不会执行。
    # with open(chunks_json_path, "w", encoding="utf-8") as f:
    # json.dump(chunks, f, ensure_ascii=False, indent=4)

def test_refine_chunks_demo():
    sections = [
        {
            "title": "## 功能说明",
            "content": "## 功能说明\n\n" + "A" * 1200,
            "file_title": "demo",
        },
        {
            "title": "## 参数说明",
            "content": "## 参数说明\n\n参数1：xxx",
            "file_title": "demo",
        },
    ]
    result = refine_chunks(sections, max_len=1000, min_len=600)
    print(result)
test_refine_chunks_demo()