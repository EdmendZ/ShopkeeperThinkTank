"""构造 Milvus filter expression 时使用的字符串字面量转义工具。"""


def escape_milvus_string(value: str) -> str:
    """将值转换为可嵌入双引号 Milvus filter 的单行字符串。

    反斜杠和双引号按 expression literal 规则转义；CR、LF 与 Tab 被替换为空格，
    避免用户输入改变 filter 的行结构。为兼容历史调用，运行时收到 ``None`` 时返回
    空字符串，其他非字符串值通过 ``str`` 规范化。

    Args:
        value: 商品主体、文件标题等需要放入 filter expression 的值。

    Returns:
        已转义且不含控制换行符的字符串。
    """
    if value is None:
        return ""
    s = str(value)
    # 必须先处理反斜杠，否则后续插入的 escape character 会被再次转义。
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return s
