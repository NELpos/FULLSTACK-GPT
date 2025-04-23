import streamlit as st

st.set_page_config(page_title="FullstackGPT Home", page_icon="🧊")

st.title("FullstackGPT Home")
with st.sidebar:
    st.title("Sidebar title")
    st.text_input("xxxx")

tab_one, tab_two, tab_three = st.tabs(["A", "B", "C"])

with tab_one:
    st.write("A")

with tab_two:
    st.write("B")

with tab_three:
    st.write("C")
# today = datetime.today().strftime("%H:%M:%S")

# st.title(today)

# model = st.selectbox("Choose your model", ("GsPT-3", "GPT-4"))
# st.write(model)

# if model == "GPT-3":
#     st.write("cheap")
# else:
#     st.write("not cheap")

# name = st.text_input("Wgat is your name?")
# st.write(name)

# value = st.slider(
#     "temperature",
#     min_value=0.1,
#     max_value=1.0,
# )
# st.write(value)
