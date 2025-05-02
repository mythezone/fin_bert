# 使用 Python 3.13 官方镜像
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 将当前目录的所有内容复制到容器中
COPY . .

# 使用 pip 安装依赖（最好先升级 pip）
RUN pip install --upgrade pip && \
    pip install pandas python-dotenv sqlalchemy pymysql plotly gradio

# 默认执行命令（也可以在 docker-compose 中覆盖）
CMD ["python", "web/main.py"]