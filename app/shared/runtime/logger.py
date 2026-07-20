"""
项目日志工具类
基于loguru实现，支持.env配置控制台/文件双输出，自动生成logs/app_年月日.log
特性：
1. 配置驱动：通过.env开关输出、修改日志级别
2. 自动路径：文件日志默认输出到 项目根/logs/app_YYYYMMDD.log
3. 自动清理：按配置保留日志，自动删除过期文件
4. 中文友好：utf-8编码，彻底解决中文乱码
5. 异步安全：开启异步入队，支持多线程/异步场景，避免日志错乱
6. 开箱即用：项目所有模块直接导入logger即可使用
7. 位置终极精准：穿透loguru内部+工具类自身，完美显示业务模块实际调用位置
"""
import sys
import inspect
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger


# -------------------------- 第一步：加载 .env 配置文件 --------------------------
# 将项目环境变量写入当前进程；未配置的项目会继续使用下方定义的默认值。
load_dotenv()

# -------------------------- 第二步：读取 .env 配置（带默认值，防止配置缺失） --------------------------
# 布尔开关仅在值（忽略大小写）为 "true" 时开启；日志等级统一转为大写以匹配 Loguru 的等级名称。
LOG_CONSOLE_ENABLE = os.getenv("LOG_CONSOLE_ENABLE", "True").lower() == "true"
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
LOG_FILE_ENABLE = os.getenv("LOG_FILE_ENABLE", "True").lower() == "true"
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()
LOG_FILE_RETENTION = os.getenv("LOG_FILE_RETENTION", "7 days")

# -------------------------- 第三步：定义日志路径（自动推导项目根） --------------------------
# logger.py 位于 app/shared/runtime；从当前文件的绝对路径向上四级得到项目根目录，不依赖启动命令所在目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
# {time:YYYYMMDD} 是 Loguru 在写入时解析的日期模板，LOG_FILE_PATH 并非固定某一天的实际文件名。
LOG_FILE_NAME = "app_{time:YYYYMMDD}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME

# -------------------------- 第四步：定义日志格式（彩色、结构化、易读） --------------------------
# 依次输出时间、等级、模块名、函数名、行号和消息；<green>/<cyan>/<level> 为 Loguru 的终端颜色标记。
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <20}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: <4}</cyan> - "
    "<level>{message}</level>"
)

# -------------------------- 第五步：初始化日志配置（核心方法） --------------------------
def init_logger():
    """
    初始化全局日志配置
    1. 移除loguru默认控制台输出（避免重复打印）
    2. 根据.env配置开启/关闭控制台输出
    3. 根据.env配置开启/关闭文件输出（自动创建logs文件夹）
    4. 配置日志格式、级别、分割、保留策略
    :return: 配置完成的loguru logger实例
    """
    # 1. 移除loguru默认的控制台输出
    logger.remove()

    # 2. 配置控制台输出（若 .env 开启）；控制台与文件可设置不同的最低输出等级。
    if LOG_CONSOLE_ENABLE:
        logger.add(
            sink=sys.stdout,
            level=LOG_CONSOLE_LEVEL,
            format=LOG_FORMAT,
            colorize=True,
            # 通过队列串行写入，适用于多线程和异步场景，避免日志交错。
            enqueue=True
        )

    # 3. 配置文件输出（若 .env 开启）；目录不存在时自动创建。
    if LOG_FILE_ENABLE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=LOG_FILE_PATH,
            level=LOG_FILE_LEVEL,
            format=LOG_FORMAT,
            # 每日零点创建新日志文件，并依据 .env 的保留策略清理过期日志。
            rotation="00:00",
            retention=LOG_FILE_RETENTION,
            # 使用 UTF-8 避免中文乱码；队列写入避免并发场景下文件内容交错。
            encoding="utf-8",
            enqueue=True,
            # 异常日志附带增强的调用栈和变量诊断信息，便于排查问题。
            backtrace=True,
            diagnose=True
        )

    return logger

# -------------------------- 第六步：初始化并修正全局 logger 的调用位置 --------------------------
base_logger = init_logger()


def fix_log_position(record):
    """将日志显示的位置修正为业务调用方，而不是 Loguru 或当前封装模块内部。"""
    # 从最近调用栈开始，跳过 Loguru 内部帧及本文件自身帧，定位第一处业务代码。
    for frame in inspect.stack():
        if ("_logger.py" in frame.filename or frame.function == "_log") or "logger.py" in frame.filename:
            continue
        # 同时处理 Windows（反斜杠）与 POSIX（正斜杠）路径，提取仅用于日志展示的文件名。
        record.update(
            name=frame.filename.split("/")[-1].split("\\")[-1],
            function=frame.function,
            line=frame.lineno
        )
        break


# patch 会在每条日志生成前执行位置修正；导出的 logger 可供业务模块直接导入使用。
logger = base_logger.patch(fix_log_position)


from functools import wraps
import time
from typing import Mapping

def _trace_id(state) -> str:
    """从状态映射中按 session_id、task_id 的优先级提取追踪 ID，缺失时返回 "-"。"""
    if isinstance(state, Mapping):
        return str(state.get("session_id") or state.get("task_id") or "-")
    return "-"


def node_log(node_name: str):
    """为首个参数为 state 的节点函数记录开始、结束、耗时和异常日志。"""
    def deco(func):
        """接收被装饰节点函数，并返回增加日志行为后的包装函数。"""

        # 保留被装饰函数的名称、文档等元数据，便于调试和框架识别。
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            """执行原节点函数，同时记录追踪 ID、耗时和异常堆栈。"""

            trace_id = _trace_id(state)
            start_ts = time.time()
            logger.info(f"[{node_name}] 节点开始，追踪ID={trace_id}")
            try:
                # 耗时仅覆盖实际业务函数执行；异常记录后会继续抛出，不改变原有业务语义。
                result = func(state, *args, **kwargs)
                cost_ms = int((time.time() - start_ts) * 1000)
                logger.info(f"[{node_name}] 节点完成，追踪ID={trace_id}，耗时={cost_ms}ms")
                return result
            except Exception:
                logger.exception(f"[{node_name}] 节点异常，追踪ID={trace_id}")
                raise
        return wrapper
    return deco


def step_log(step_name: str):
    """
    步骤日志装饰器：适用于普通函数，不要求首个参数为 state。
    - 自动打印步骤开始 / 步骤完成 / 步骤异常（含堆栈）
    - 不吞异常，保持原有业务语义
    """
    def deco(func):
        """接收任意业务函数，并返回带步骤日志的包装函数。"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            """透传全部参数执行原函数，并在异常后重新抛出。"""

            start_ts = time.time()
            logger.info(f"[{step_name}] 步骤开始")
            try:
                result = func(*args, **kwargs)
                cost_ms = int((time.time() - start_ts) * 1000)
                logger.info(f"[{step_name}] 步骤完成，耗时={cost_ms}ms")
                return result
            except Exception:
                logger.exception(f"[{step_name}] 步骤异常")
                raise
        return wrapper
    return deco

# -------------------------- 测试代码（仅直接运行本文件时执行） --------------------------
# 用于人工验证日志等级、输出路径及异常堆栈，不属于业务初始化流程。
if __name__ == '__main__':
    # 【debug】开发调试用，记录细节、变量，上线一般关闭
    logger.debug("【调试】进入主程序入口，开始初始化日志")

    # 【info】正常流程日志，记录程序运行状态
    logger.info("【信息】logger.py内部调用（仅测试，业务模块调用会显示正确文件名）")

    print(f"日志文件输出路径：{LOG_FILE_PATH}")

    # 【warning】警告，不影响运行，但需要关注
    logger.warning("【警告】未读取到自定义配置，使用默认配置")

    # 【error】当前功能出错，程序不会崩溃，但业务失败
    logger.error("【错误】logger.py内部调用（仅测试，业务模块调用会显示正确文件名）")

    # 【exception】必须在except里，自动打印完整异常堆栈（定位bug用）
    try:
        result = 10 / 0
        logger.info(f"【信息】业务计算结果：{result}")
    except Exception:
        logger.exception("【异常】捕获到业务异常，输出完整堆栈信息")
