import os
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from sqlalchemy import inspect, text
from dotenv import load_dotenv
from data.mysql import create_mysql_engine

# Load DB config
load_dotenv()

TMP_DIR = "./data/tmp"
engine = create_mysql_engine()


# 获取所有股票表名（假设都是以某种规则命名的，例如全部为小写或带数字）
def get_stock_table_names():
    inspector = inspect(engine)
    return sorted(inspector.get_table_names())


# 主查询与绘图逻辑
def show_candlestick(stock_table, date_str):
    query = text(
        f"""
        SELECT * FROM `{stock_table}` 
        WHERE DATE(datetime) = :date 
        ORDER BY datetime ASC
    """
    )
    df = pd.read_sql(query, engine, params={"date": date_str}, parse_dates=["datetime"])

    if df.empty:
        return f"没有找到 {stock_table} 在 {date_str} 的数据", None

    # 保存数据为 CSV
    file_name = f"{stock_table}-{date_str}-1m.csv"
    file_path = os.path.join(TMP_DIR, file_name)
    df.to_csv(file_path, index=False)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["datetime"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        ]
    )
    fig.update_layout(
        title=f"{stock_table} - {date_str} OHLC",
        xaxis_title="Time",
        yaxis_title="Price",
    )
    return "", fig, file_path


# 构建 Gradio 界面
def build_ui():
    stock_list = get_stock_table_names()

    with gr.Blocks() as demo:
        gr.Markdown("### 📈 股票分钟级 OHLC 蜡烛图可视化")

        with gr.Row():
            stock_dropdown = gr.Dropdown(choices=stock_list, label="选择股票（表名）")
            date_picker = gr.Textbox(
                label="输入日期 (格式: YYYY-MM-DD)", placeholder="如：2014-04-23"
            )
        btn = gr.Button("生成图表并导出数据")
        msg_output = gr.Textbox(label="提示信息", interactive=False)

        plot_output = gr.Plot()
        file_output = gr.File(label="点击下载当天数据", interactive=False)
        btn.click(
            show_candlestick,
            inputs=[stock_dropdown, date_picker],
            outputs=[msg_output, plot_output, file_output],
        )

    return demo


# 启动
if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
