# 使用一个基础镜像
FROM python:3.8

# 安装OpenGL库
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y python3-opencv

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 设置工作目录
WORKDIR $HOME/app
# 复制根目录内的所有内容到容器的工作目录
COPY --chown=user . $HOME/app

# 安装Python依赖项
RUN pip install -r requirements.txt

# 暴露7860端口
EXPOSE 7860

# 运行Python应用程序
CMD ["python", "app.py"]
