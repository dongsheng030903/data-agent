import os

# Windows 下部分环境会把 IPv6 地址（如 [::1]）写进 NO_PROXY，
# 导致 httpx2 在创建 OpenAI 客户端时解析失败。这里统一成安全的本地地址。
_no_proxy = os.environ.get("NO_PROXY", "") or os.environ.get("no_proxy", "")
if "[" in _no_proxy or "]" in _no_proxy:
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    os.environ["no_proxy"] = "localhost,127.0.0.1"
