FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# 换 apt 源
RUN sed -i 's@archive.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential git vim && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        -r /tmp/requirements.txt

# 拷贝代码
WORKDIR /workspace
COPY . /workspace

ENTRYPOINT ["python", "train.py"]