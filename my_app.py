import streamlit as st
import pandas as pd
import numpy as np

# st.title() 會在網頁上顯示一個大標題
st.title('Streamlit 入門範例 🚀')

# st.write() 可以用來顯示文字、數字、DataFrame 等等
st.write("這是一個非常簡單的互動式應用！")

# 建立一個滑桿 (slider) 小工具
# 使用者可以拖動它來選擇 1 到 50 之間的一個數字
# 'x' 是儲存滑桿數值的變數
x = st.slider('請選擇一個數字來決定圖表的行數：', 1, 50, 10)

# 顯示使用者選擇的數字
st.write(f"您選擇了：{x}")

# 根據使用者選擇的數字 x，建立一個隨機的 DataFrame
# 它有 x 行和 2 列
chart_data = pd.DataFrame(
    np.random.randn(x, 2),
    columns=['a', 'b'])

# st.line_chart() 會根據傳入的 DataFrame 畫出一個折線圖
st.line_chart(chart_data)

st.write("看到沒？拖動上面的滑桿，下面的圖表就會即時更新！")