from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import os

app = Flask(__name__)

# Параметры: путь к твоему CSV
CSV_PATH = os.path.join("data", "dataset.csv")

def load_and_prepare():
    # Если файл ещё не скачан, можно сюда добавить скачивание с Kaggle API или HTTP
    df = pd.read_csv(CSV_PATH)

    # Пример: пусть датасет имеет колонки "Time" и "Value"
    # Если твой датасет другой — поправь эти колонки
    X = df[["Time"]]
    y = df["Value"]

    model = LinearRegression()
    model.fit(X, y)
    df["Predicted"] = model.predict(X)

    return df

def make_plot(df):
    fig = px.scatter(df, x="Time", y="Value", title="Линейная регрессия: Time vs Value")
    fig.add_scatter(x=df["Time"], y=df["Predicted"], mode="lines", name="Regression Line")

    # Сохраним как HTML
    html_file = os.path.join("static", "plot.html")
    fig.write_html(html_file)
    return "static/plot.html"

@app.route("/")
def index():
    df = load_and_prepare()
    plot_path = make_plot(df)
    return render_template("index.html", plot_path=plot_path)

if __name__ == "__main__":
    app.run(debug=True)