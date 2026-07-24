"""Markdown 图片增强服务：定位正文引用的图片，生成摘要、上传 MinIO 并替换链接。"""

import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Dict

from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.rag.import_.config import SUPPORTED_IMAGE_EXTENSIONS
from app.infra.object_storage import minio_gateway
from app.shared.utils.rate_limit_utils import apply_api_rate_limit

@step_log("enrich_markdown_images")
def enrich_markdown_images(state: dict) -> dict:
    """串联读取、扫描、图片总结、上传替换和 Markdown 备份五个步骤。

    状态中至少需要 ``md_path``；函数会补充或更新 ``md_content`` 与 ``md_path``。
    图片目录不存在或为空时直接返回，不调用视觉模型和对象存储。
    """

    md_content, md_path_obj, images_path_obj = load_markdown_and_image_dir(state)
    if not images_path_obj.is_dir() or len(list(images_path_obj.iterdir())) == 0:
        logger.warning("图片文件夹为空或者没有图片,无需后续处理!")
        return state

    image_context_list = scan_images(md_content, images_path_obj)
    logger.info(f"已经获取上下文信息:{image_context_list}")
    image_summaries_dict = summarize_images(image_context_list, md_path_obj.stem)
    new_md_content = upload_images_and_replace(image_context_list, image_summaries_dict, md_content, md_path_obj.stem)
    state["md_content"] = new_md_content
    state["md_path"] = backup_markdown(new_md_content, md_path_obj)
    return state

@step_log("load_markdown_and_image_dir")
def load_markdown_and_image_dir(state: dict) -> tuple[str, Path, Path]:
    """取得 Markdown 正文、文件路径对象及同级 ``images`` 目录路径。

    如果状态里没有缓存正文，就按 ``md_path`` 从磁盘读取，并把内容回写到状态。
    返回类型 ``tuple[str, Path, Path]`` 表示一个固定三项元组。
    """

    md_content = state.get("md_content", "")
    md_path = state.get("md_path")
    if not md_path:
        logger.error("md_path核心参数为空,无法继续!!")
        raise ValueError("md_path核心参数为空,无法继续!!")

    md_path_obj = Path(md_path)
    if not md_content:
        logger.warning("md_content为空,根据地址读取!")
        md_content = md_path_obj.read_text(encoding="utf-8")
        state["md_content"] = md_content

    images_path_obj = md_path_obj.parent / "images"
    return md_content, md_path_obj, images_path_obj

@step_log("is_supported_image")
def is_supported_image(filename: str) -> bool:
    """按小写扩展名判断文件是否属于项目支持的图片格式。"""

    # splitext 返回 (不含扩展名的部分, 扩展名)，[1] 取扩展名；lower 兼容 .PNG 等写法。
    return os.path.splitext(filename)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS


@step_log("scan_images")
def scan_images(md_content: str, images_path_obj: Path) -> list[tuple[str, str, tuple[str, str]]]:
    """
    扫描图片目录 + Markdown 内容，找出【真正被MD引用的图片】，并提取图片上下文
    作用：只处理真正用到的图片，过滤无效文件，同时截取上下文给视觉模型做摘要

    :param md_content: Markdown 文本内容
    :param images_path_obj: 图片所在文件夹路径
    :return: 列表 -> (图片名, 图片完整路径, (上文100字符, 下文100字符))
    """
    # 存储最终筛选出的【有效图片 + 上下文信息】
    image_context_list: list[tuple[str, str, tuple[str, str]]] = []
    # 遍历图片目录下的所有文件
    for image_file in images_path_obj.iterdir():
        image_name = image_file.name
        # 1. 过滤：不是支持的图片格式直接跳过
        if not is_supported_image(image_name):
            logger.warning(f"{image_name}不是图片,无需处理,跳过本次!!")
            continue
        # 正则含义：\! 和 \[ 匹配 Markdown 开头的字面量 "!["；.*? 是非贪婪任意内容；
        # re.escape 把文件名里的 .、(、[ 等正则特殊字符转义成普通字符。
        rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        # search 返回第一个 Match 对象；没有匹配时返回 None。
        match_obj = rep.search(md_content)
        # 图片存在，但 MD 里没用到 → 跳过
        if not match_obj:
            logger.warning(f"{image_name}没有在md中使用,跳过本次处理!")
            continue
        # span() 返回 (start, end)，start 包含在匹配内，end 是匹配结束后的下标。
        start, end = match_obj.span()
        # 4. 截取图片【上方100字符】作为上文（防止越界）
        pre_context = md_content[max(start - 100, 0):start]
        # 5. 截取图片【下方100字符】作为下文（防止越界）
        pos_context = md_content[end:min(end + 100, len(md_content))]
        # 6. 把有效信息加入结果列表
        # 格式：(图片名, 图片完整路径, (上文, 下文))
        image_context_list.append((image_name, str(image_file), (pre_context, pos_context)))

    # 返回所有真正被使用的图片信息
    return image_context_list
@step_log("summarize_images")
def summarize_images(image_context_list: list[tuple[str, str, tuple[str, str]]], stem: str) -> Dict[str, str]:
    """逐张调用视觉模型生成摘要，返回“图片文件名 -> 摘要”的字典。

    图片二进制先编码为 Base64，再放入 ``data:`` URL；每次调用前通过限流函数
    控制请求频率，减少触发模型服务限额的概率。
    """

    image_summaries_dict: Dict[str, str] = {}
    vm_model = llm_provider.vision_chat()
    for image_name, image_path_str, context in image_context_list:
        apply_api_rate_limit()
        image_context_prompt = load_prompt("image_summary", root_folder=stem, image_content=context)
        image_path_obj = Path(image_path_str)
        # b64encode 的输入和输出都是 bytes；decode("utf-8") 把 Base64 的 ASCII 字节
        # 变成 str，才能拼进 JSON/消息里的 data URL。它不是在用 UTF-8 解码原始图片。
        image_data = base64.b64encode(image_path_obj.read_bytes()).decode(encoding="utf-8")
        # HumanMessage 的 content 使用列表时可同时携带图片块和文本提示词块。
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mimetypes.guess_type(image_name)[0]};base64,{image_data}"
                    },
                },
                {"type": "text", "text": image_context_prompt},
            ]
        )
        # | 把视觉模型与字符串解析器连接成 LangChain 管道，invoke 执行整条管道。
        summary = (vm_model | StrOutputParser()).invoke([message])
        image_summaries_dict[image_name] = summary
    return image_summaries_dict

@step_log("upload_images_and_replace")
def upload_images_and_replace(
    image_context_list: list[tuple[str, str, tuple[str, str]]],
    image_summaries_dict: Dict[str, str],
    md_content: str,
    stem: str,
) -> str:
    """
    图片上传 + Markdown内容替换
    1. 清空MinIO中该文档的旧图片（避免脏数据）
    2. 上传新图片到MinIO
    3. 将MD中原生本地图片 → 替换为【图片摘要】+【在线URL】
    返回替换完成后的新MD文本
    """
    # 获取MinIO客户端实例
    minio_client = minio_gateway.client()

    # 以文档 stem 作为 object prefix；先清理旧对象，避免重复导入留下孤儿图片。
    # 列出当前文档(stem)在MinIO中已存在的所有图片
    object_list = minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        prefix=f"{minio_gateway.image_dir[1:]}/{stem}/",
        recursive=True,
    )
    # 构造批量删除对象列表
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    # 执行批量删除
    errors = minio_client.remove_objects(
        bucket_name=minio_gateway.bucket_name,
        delete_object_list=delete_object_list,
    )
    # 打印删除失败的错误信息
    for error in errors:
        logger.warning(f"删除失败,失败原因:{error}")

    # 上传成功后记录 URL，后续只替换实际出现在 Markdown 中的引用。
    # 存储：图片文件名 → 在线访问URL
    image_url_dict: dict[str, str] = {}
    for image_name, image_path_str, _ in image_context_list:
        try:
            # 上传本地图片到MinIO
            minio_client.fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=f"{minio_gateway.image_dir}/{stem}/{image_name}",
                file_path=image_path_str,
                content_type=mimetypes.guess_type(image_name)[0],
            )
            # 构建图片在线URL并保存
            image_url_dict[image_name] = minio_gateway.build_image_url(stem=stem, image_name=image_name)
        except Exception:
            logger.warning(f"本次图片上传失败:{image_name},跳过继续上传下一张!")
            continue

    # 如果所有图片都上传失败，直接返回原内容
    if not image_url_dict:
        logger.warning("图片上传全部失败!")
        return md_content

    # replacement 同时加入视觉摘要，令纯文本检索也能利用图片语义。
    # 遍历所有已上传成功的图片
    for image_name, image_url in image_url_dict.items():
        # 拿到当前图片的AI摘要
        image_summary = image_summaries_dict[image_name]
        # 正则匹配：![xxx](xxx/图片名)
        rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")

        # ----------------------- 重点：为什么用 lambda？-----------------------
        # re.sub 的第二个参数需要是【替换模板】或【处理函数】
        # 我们需要动态拼接：![图片摘要](在线URL) → 必须用函数动态生成
        # lambda _: ...  这里的 _ 表示匹配到的对象（我们不需要它，所以用下划线忽略）
        # -------------------------------------------------------------------
        md_content = rep.sub(lambda _: f"![{image_summary}]({image_url})", md_content)

    # 返回替换完成的最终MD内容
    return md_content

@step_log("backup_markdown")
def backup_markdown(new_md_content: str, md_path_obj: Path) -> str:
    """
    备份并保存【增强后的新 Markdown 文件】
    作用：不覆盖原始 MD 文件，生成一个 _new.md 的新版本，保证原始文件安全
    :param new_md_content: 图片增强、替换完成后的最新 MD 内容
    :param md_path_obj: 原始 MD 文件路径对象
    :return: 新 MD 文件的字符串路径
    """
    # 拼接新文件路径：原始文件名 + _new.md（例如：手册.md → 手册_new.md）
    new_md_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")

    # 将新的 Markdown 内容写入文件，使用 UTF-8 编码保证中文不乱码
    new_md_path_obj.write_text(new_md_content, encoding="utf-8")

    # 返回新文件的字符串路径，存入 state 供后续节点使用
    return str(new_md_path_obj)
