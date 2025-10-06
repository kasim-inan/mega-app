import time
import datetime

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

st.set_page_config("Anthropic dark theme", "🟤", layout="wide")
st.logo(
    "https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/anthropic.png"
)
st.title("Anthropic dark theme")

if st.checkbox("Enable CSS hacks", True):
    codeBackgroundColor = "#232322"

    titleFontSize = "40px"
    titleFontWeight = "500"
    headerFontSize = "32px"
    headerFontWeight = "500"
    subheaderFontSize = "24px"
    subheaderFontWeight = "500"

    pageHoverBackgroundColor = "#1a1918"
    pageFontSize = "14px"

    activePageBackgroundColor = "#1a1918"
    activePageHoverBackgroundColor = "#1a1918"
    
    
    st.html(
        f"""
        <style>
        body {{
            -webkit-font-smoothing: antialiased;
        }}
        
        .stSidebar > div:last-of-type > div > div {{
            background-image: linear-gradient(to right, transparent 20%, rgba(34, 34, 34, 0.3) 28%, transparent 36%);
        }}
        
        .stCode pre {{
            background-color: {codeBackgroundColor};
        }}
        
        h1 {{
            font-size: {titleFontSize} !important;
            font-weight: {titleFontWeight} !important;
        }}
        
        h2 {{
            font-size: {headerFontSize} !important;
            font-weight: {headerFontWeight} !important;
        }}
        
        h3 {{
            font-size: {subheaderFontSize} !important;
            font-weight: {subheaderFontWeight} !important;
        }}
        
        /* First page in sidebar nav */
        [data-testid="stSidebarNav"] li:first-of-type a {{
            background-color: {activePageBackgroundColor} !important;
        }}
        [data-testid="stSidebarNav"] li:first-of-type a:hover {{
            background-color: {activePageHoverBackgroundColor} !important;
        }}
        
        /* Other pages in sidebar nav */
        [data-testid="stSidebarNav"] li a:hover {{
            background-color: {pageHoverBackgroundColor} !important;
        }}
        [data-testid="stSidebarNav"] li a span {{
            font-size: {pageFontSize} !important;
        }}
        </style>
        """
    )


def page1():
    pass


def page2():
    pass


def page3():
    pass


st.navigation(
    {
        "General": [
            st.Page(page1, title="Home", icon=":material/home:"),
            st.Page(page2, title="Data visualizations", icon=":material/monitoring:"),
        ],
        "Admin": [st.Page(page3, title="Settings", icon=":material/settings:")],
    }
)


"## Write and magic"
st.write("st.write")
"magic"


"## Text elements"
st.markdown("st.markdown")
st.markdown("st.markdown with help", help="Hello!")
st.markdown(
    "Markdown features: **bold** *italic* ~strikethrough~ [link](https://streamlit.io) `code` $a=b$ 🐶 :cat: :material/home: :streamlit: <- -> <-> -- >= <= ~="
)
st.markdown("""
Text colors: 

:blue[blue] :green[green] :orange[orange] :red[red] :violet[violet] :gray[gray] :rainbow[rainbow] :primary[primary]

:blue-background[blue] :green-background[green] :orange-background[orange] :red-background[red] :violet-background[violet] :gray-background[gray] :rainbow-background[rainbow] :primary-background[primary]

:blue-background[:blue[blue]] :green-background[:green[green]] :orange-background[:orange[orange]] :red-background[:red[red]] :violet-background[:violet[violet]] :gray-background[:gray[gray]] :rainbow-background[:rainbow[rainbow]] :primary-background[:primary[primary]]
""")
st.title("st.title")
st.title("st.title with help", help="Hello!")
st.header("st.header")
st.header("st.header with help", help="Hello!")
st.header("st.header with blue divider", divider="blue")
st.header("st.header with green divider", divider="green")
st.header("st.header with orange divider", divider="orange")
st.header("st.header with red divider", divider="red")
st.header("st.header with violet divider", divider="violet")
st.header("st.header with gray divider", divider="gray")
st.header("st.header with rainbow divider", divider="rainbow")
st.subheader("st.subheader")
st.subheader("st.subheader with help", help="Hello!")
st.caption("st.caption")
st.caption("st.caption with help", help="Hello!")
st.code("# st.code\na = 1234")
st.code("# st.code with line numbers\na = 1234", line_numbers=True)
st.code(
    '# st.code with line wrapping\na = "This is a very very very very very very very very very very very very long string"',
    wrap_lines=True,
)
# with st.echo():
#     st.write("st.echo")
st.latex(r"\int a x^2 \,dx")
st.latex(r"\int a x^2 \,dx", help="Hello!")
st.text("st.text")
st.text("st.text with help", help="Hello!")
st.divider()


"## Data elements"
np.random.seed(42)
data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

"st.dataframe"
st.dataframe(data)

"st.dataframe with column configuration"
st.dataframe(
    data,
    column_config={
        "a": st.column_config.NumberColumn("Column A", format="%.3f"),
        "b": st.column_config.NumberColumn("Column B", format="%.3f"),
        "c": st.column_config.ProgressColumn("Progress", min_value=-3, max_value=3),
    },
    hide_index=False,
)

"st.data_editor"
st.data_editor(data)

"st.column_config"
data_df = pd.DataFrame(
    {
        "column": ["foo", "bar", "baz"],
        "text": ["foo", "bar", "baz"],
        "number": [1, 2, 3],
        "checkbox": [True, False, True],
        "selectbox": ["foo", "bar", "foo"],
        "datetime": pd.to_datetime(
            ["2021-01-01 00:00:00", "2021-01-02 00:00:00", "2021-01-03 00:00:00"]
        ),
        "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
        "time": pd.to_datetime(["00:00:00", "01:00:00", "02:00:00"]),
        "list": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "link": [
            "https://streamlit.io",
            "https://streamlit.io",
            "https://streamlit.io",
        ],
        "image": [
            "https://picsum.photos/200/300",
            "https://picsum.photos/200/300",
            "https://picsum.photos/200/300",
        ],
        "area_chart": [[1, 2, 1], [2, 3, 1], [3, 1, 2]],
        "line_chart": [[1, 2, 1], [2, 3, 1], [3, 1, 2]],
        "bar_chart": [[1, 2, 1], [2, 3, 1], [3, 1, 2]],
        "progress": [0.1, 0.2, 0.3],
    }
)

st.data_editor(
    data_df,
    column_config={
        "column": st.column_config.Column(
            "Column", help="A column tooltip", pinned=True
        ),
        "text": st.column_config.TextColumn("TextColumn"),
        "number": st.column_config.NumberColumn("NumberColumn"),
        "checkbox": st.column_config.CheckboxColumn("CheckboxColumn"),
        "selectbox": st.column_config.SelectboxColumn(
            "SelectboxColumn", options=["foo", "bar", "baz"]
        ),
        "datetime": st.column_config.DatetimeColumn("DatetimeColumn"),
        "date": st.column_config.DateColumn("DateColumn"),
        "time": st.column_config.TimeColumn("TimeColumn"),
        "list": st.column_config.ListColumn("ListColumn"),
        "link": st.column_config.LinkColumn("LinkColumn"),
        "image": st.column_config.ImageColumn("ImageColumn"),
        "area_chart": st.column_config.AreaChartColumn("AreaChartColumn"),
        "line_chart": st.column_config.LineChartColumn("LineChartColumn"),
        "bar_chart": st.column_config.BarChartColumn("BarChartColumn"),
        "progress": st.column_config.ProgressColumn("ProgressColumn"),
    },
)

"st.table"
st.table(data.iloc[0:5])

col1, col2 = st.columns(2)
col1.metric("st.metric positive", 42, 2)
col2.metric("st.metric negative", 42, -2)

col1, col2 = st.columns(2)
col1.metric("st.metric with border positive", 42, 2, border=True)
col2.metric("st.metric with border negative", 42, -2, border=True)

"st.json"
st.json(
    {
        "foo": "bar",
        "numbers": [
            123,
            4.56,
        ],
        "level1": {"level2": {"level3": {"a": "b"}}},
    },
    expanded=2,
)


"## Chart elements"
data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
"st.area_chart"
stack = st.radio(
    "stack",
    [None, True, False, "normalize", "center"],
    horizontal=True,
    key="area_chart_stack",
)
st.area_chart(data, x_label="x label", y_label="y label", stack=stack)
"st.bar_chart"
horizontal = st.toggle("horizontal", False)
stack = st.radio(
    "stack",
    [None, True, False, "normalize", "center"],
    horizontal=True,
    key="bar_chart_stack",
)
st.bar_chart(
    data, x_label="x label", y_label="y label", horizontal=horizontal, stack=stack
)
"st.line_chart"
st.line_chart(data, x_label="x label", y_label="y label")
"st.scatter_chart"
st.scatter_chart(data, x_label="x label", y_label="y label")

"st.map"
df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=["lat", "lon"]
)
st.map(df)

"st.pyplot"
fig, ax = plt.subplots()
ax.hist(data, bins=20)
st.pyplot(fig)

"st.pyplot - Multiple subplots"
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(data['a'], 'r-')
axes[0, 0].set_title('Line Plot')
axes[0, 1].scatter(data['a'], data['b'])
axes[0, 1].set_title('Scatter Plot')
axes[1, 0].bar(range(len(data['c'][:10])), data['c'][:10])
axes[1, 0].set_title('Bar Chart')
axes[1, 1].boxplot([data['a'], data['b'], data['c']])
axes[1, 1].set_title('Box Plot')
plt.tight_layout()
st.pyplot(fig)

"st.altair_chart"
st.altair_chart(
    alt.Chart(data)
    .mark_circle()
    .encode(x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"]),
    use_container_width=True,
)

"st.vega_lite_chart"
st.vega_lite_chart(
    data,
    {
        "mark": {"type": "circle", "tooltip": True},
        "encoding": {
            "x": {"field": "a", "type": "quantitative"},
            "y": {"field": "b", "type": "quantitative"},
            "size": {"field": "c", "type": "quantitative"},
            "color": {"field": "c", "type": "quantitative"},
        },
    },
    use_container_width=True,
)

"st.plotly_chart - Gapminder"
df = px.data.gapminder()
fig = px.scatter(
    df.query("year==2007"),
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    log_x=True,
    size_max=60,
)
st.plotly_chart(fig, use_container_width=True)

"st.plotly_chart - 3D Surface"
z_data = np.random.randn(50, 50)
fig = go.Figure(data=[go.Surface(z=z_data)])
fig.update_layout(title='3D Surface Plot', autosize=True)
st.plotly_chart(fig, use_container_width=True)

"st.plotly_chart - Heatmap"
heatmap_data = np.random.randn(20, 20)
fig = go.Figure(data=go.Heatmap(z=heatmap_data, colorscale='Viridis'))
fig.update_layout(title='Heatmap')
st.plotly_chart(fig, use_container_width=True)

"st.bokeh_chart"
if st.toggle("Show Bokeh chart (has some issues)", False):
    from bokeh.plotting import figure

    x = [1, 2, 3, 4, 5]
    y = [6, 7, 2, 4, 5]
    p = figure(title="simple line example", x_axis_label="x", y_axis_label="y")
    p.line(x, y, legend_label="Trend", line_width=2)
    st.bokeh_chart(p, use_container_width=True)

"st.pydeck_chart"
data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=["lat", "lon"]
)
st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=37.76,
            longitude=-122.4,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position="[lon, lat]",
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=data,
                get_position="[lon, lat]",
                get_color="[200, 30, 0, 160]",
                get_radius=200,
            ),
        ],
    )
)

"st.graphviz_chart"
st.graphviz_chart(
    """
    digraph {
        run -> intr
        intr -> runbl
        runbl -> run
        run -> kernel
        kernel -> zombie
        kernel -> sleep
        kernel -> runmem
        sleep -> swap
        swap -> runswap
        runswap -> new
        runswap -> runmem
        new -> runmem
        sleep -> runmem
    }
    """
)


"## Input widgets"
if st.button("st.button"):
    st.write("You pressed the button!")

if st.button("st.button primary", type="primary"):
    st.write("You pressed the button!")

if st.button("st.button tertiary", type="tertiary"):
    st.write("You pressed the button!")

if st.button("st.button with icon", icon=":material/home:"):
    st.write("You pressed the button!")

if st.button("st.button disabled", disabled=True):
    st.write("You pressed the button!")

text_contents = "This is some text"
st.download_button("st.download_button", data=text_contents)

"st.feedback"
st.feedback("thumbs")
st.feedback("faces")
st.feedback("stars")

st.link_button("st.link_button", "https://streamlit.io")

st.page_link("https://streamlit.io", label="st.page_link", icon=":material/home:")

checkbox_input = st.checkbox("st.checkbox", True)
st.write(f"Your checkbox input is {checkbox_input}!")

toggle_input = st.toggle("st.toggle", True)
st.write(f"Your toggle input is {toggle_input}!")

radio_input = st.radio("st.radio", ["cat", "dog"])
st.write(f"Your radio input is {radio_input}!")

radio_input = st.radio("st.radio horizontal", ["cat", "dog"], horizontal=True)
st.write(f"Your radio input is {radio_input}!")

selectbox_input = st.selectbox(
    "st.selectbox", ["cat", "dog", "monkey", "snake", "bird"]
)
st.write(f"Your selectbox input is {selectbox_input}!")

multiselect_input = st.multiselect(
    "st.multiselect",
    ["cat", "dog", "monkey", "snake", "bird"],
    default=["cat", "monkey"],
)
st.write(f"Your multiselect input is {multiselect_input}!")

pills_input = st.pills(
    "st.pills multi",
    ["cat", "dog", "monkey", "snake", "bird"],
    selection_mode="multi",
    default=["cat", "monkey"],
)
st.write(f"Your pills input is {pills_input}!")

pills_single = st.pills(
    "st.pills single",
    ["cat", "dog", "monkey", "snake", "bird"],
    selection_mode="single",
    default="cat",
)
st.write(f"Your pills input is {pills_single}!")

segmented_control_input = st.segmented_control(
    "st.segmented_control multi",
    ["cat", "dog", "monkey", "snake", "bird"],
    selection_mode="multi",
    default=["cat", "monkey"],
)
st.write(f"Your segmented control input is {segmented_control_input}!")

segmented_control_single = st.segmented_control(
    "st.segmented_control single",
    ["cat", "dog", "monkey"],
    selection_mode="single",
    default="dog",
)
st.write(f"Your segmented control input is {segmented_control_single}!")

select_slider_input = st.select_slider(
    "st.select_slider",
    options=["xsmall", "small", "medium", "large", "xlarge"],
    value="small",
)
st.write(f"Your select_slider input is {select_slider_input}!")

select_slider_range = st.select_slider(
    "st.select_slider range",
    options=["xsmall", "small", "medium", "large", "xlarge"],
    value=("small", "large"),
)
st.write(f"Your select_slider range is {select_slider_range}!")

color_input = st.color_picker("st.color_picker")
st.write(f"Your color input hex is {color_input}!")

number_input = st.number_input("st.number_input")
st.write(f"Your number input is {number_input}!")

number_input_step = st.number_input("st.number_input with step", min_value=0, max_value=100, value=50, step=5)
st.write(f"Your number input is {number_input_step}!")

slider_input = st.slider("st.slider", value=30)
st.write(f"Your slider input is {slider_input}!")

slider_range = st.slider("st.slider range", min_value=0, max_value=100, value=(25, 75))
st.write(f"Your slider range is {slider_range}!")

date_input = st.date_input("st.date_input")
st.write(f"Your date input is {date_input}!")

date_range = st.date_input(
    "st.date_input range",
    value=(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
)
st.write(f"Your date range is {date_range}!")

time_input = st.time_input("st.time_input")
st.write(f"Your time input is {time_input}!")

text_input = st.text_input("st.text_input")
st.write(f"Your text input is {text_input}!")

text_input_password = st.text_input("st.text_input password", type="password")
st.write(f"Your password has {len(text_input_password)} characters!")

text_area_input = st.text_area("st.text_area")
st.write(f"Your text_area input is {text_area_input}!")

audio_input = st.audio_input("st.audio_input")
st.write(f"Your audio input is {audio_input}!")

file_input = st.file_uploader("st.file_uploader")
st.write(f"Your file input is {file_input}!")

file_input_multiple = st.file_uploader("st.file_uploader multiple", accept_multiple_files=True)
st.write(f"You uploaded {len(file_input_multiple) if file_input_multiple else 0} files!")

if st.toggle("Show camera input (requires camera permission)", False):
    cam_input = st.camera_input("st.camera_input")
    st.write(f"Your cam input is {cam_input}!")


"## Media elements"
"st.image"
st.image("https://picsum.photos/200/300")

"st.image with caption"
st.image("https://picsum.photos/200/300", caption="Random image from Lorem Picsum")

col1, col2, col3 = st.columns(3)
with col1:
    st.image("https://picsum.photos/200/300")
with col2:
    st.image("https://picsum.photos/200/301")
with col3:
    st.image("https://picsum.photos/200/302")

"st.audio"
st.audio(
    "https://file-examples.com/wp-content/storage/2017/11/file_example_MP3_700KB.mp3"
)

"st.video"
st.video(
    "https://file-examples.com/wp-content/storage/2017/04/file_example_MP4_480_1_5MG.mp4"
)


"## Layouts and containers"

"st.columns"
a, b = st.columns(2)
a.write("column 1")
b.write("column 2")

"st.columns with different ratios"
a, b, c = st.columns([2, 1, 1])
a.write("Wide column")
b.write("Narrow column")
c.write("Narrow column")

c = st.container()
c.write("st.container")

c_border = st.container(border=True)
c_border.write("st.container with border")


@st.dialog("Test dialog")
def dialog():
    st.write("Hello there!")
    if st.button("Close"):
        st.rerun()


if st.button("Open st.dialog"):
    dialog()

a = st.empty()
a.write("st.empty")

with st.expander("st.expander"):
    st.write("works!")

with st.expander("st.expander expanded", expanded=True):
    st.write("This one starts expanded!")

with st.popover("st.popover"):
    st.write("works!")

st.sidebar.write("st.sidebar")

with st.sidebar:
    st.selectbox("st.selectbox sidebar", ["cat", "dog", "monkey", "snake", "bird"])
    st.button("st.button sidebar")
    st.checkbox("st.checkbox sidebar", True)
    st.info("st.info sidebar")
    st.expander("st.expander sidebar").write("works!")

"st.tabs"
tab_a, tab_b, tab_c = st.tabs(["tab 1", "tab 2", "tab 3"])
tab_a.write("tab 1 content")
tab_b.write("tab 2 content")
tab_c.write("tab 3 content")


"## Chat elements"

"st.chat_input"
if st.toggle("Show chat input at the bottom of the screen", False):
    st.chat_input()
else:
    st.container().chat_input()

"st.chat_message"
st.chat_message("assistant").write("Hello there!")
st.chat_message("user").write("Hi assistant!")
st.chat_message("ai").write("I'm an AI message")
st.chat_message("human").write("I'm a human message")

if st.button("Start st.status"):
    with st.status("Working on it...", expanded=True) as status:
        time.sleep(1)
        st.write("Some content...")
        time.sleep(1)
        st.write("Some content...")
        time.sleep(1)
        st.write("Some content...")
        status.update(label="Done!", state="complete")


if st.button("Start st.write_stream"):

    def stream():
        for i in ["hello", " streaming", " world"]:
            time.sleep(0.5)
            yield i

    st.write_stream(stream)


"## Status elements"
if st.button("st.progress"):
    my_bar = st.progress(0)
    for percent_complete in range(100):
        my_bar.progress(percent_complete + 1, text=f"Progress: {percent_complete + 1}%")
        time.sleep(0.05)

if st.button("st.spinner"):
    with st.spinner("Wait!"):
        time.sleep(3)
        st.write("spinner works if you saw it!")

if st.button("st.toast"):
    st.toast("Hello there!", icon="🎈")

if st.button("st.balloons"):
    st.balloons()

if st.button("st.snow"):
    st.snow()

st.success("st.success")
st.success("st.success with icon", icon=":material/home:")
st.info("st.info")
st.info("st.info with icon", icon=":material/home:")
st.warning("st.warning")
st.warning("st.warning with icon", icon=":material/home:")
st.error("st.error")
st.error("st.error with icon", icon=":material/home:")
st.exception(RuntimeError("st.exception"))


"## Execution flow"

"st.fragment"


@st.fragment
def my_fragment():
    if st.button("Wait 1s inside the fragment"):
        time.sleep(1)


my_fragment()

if st.button("st.rerun()"):
    st.rerun()

if st.button("st.stop()"):
    st.stop()
    st.write("if you see this, st.stop does not work")

with st.form(key="tester"):
    "st.form"
    text_tester = st.text_input("Your text")
    number_tester = st.number_input("Your number")
    st.form_submit_button("Submit")
st.write("Your text is:", text_tester)
st.write("Your number is:", number_tester)

with st.form(key="tester2"):
    "st.form with clear on submit"
    text_tester2 = st.text_input("Your text", key="text2")
    st.form_submit_button("Submit", on_click=lambda: None)
st.write("Your text is:", text_tester2)


st.write("## Utilities")

"st.help"
st.help(st.write)

st.write("## State Management")

"st.session_state"
if "foo" not in st.session_state:
    st.session_state["foo"] = "bar"
if "counter" not in st.session_state:
    st.session_state["counter"] = 0

st.write(st.session_state)

if st.button("Increment counter"):
    st.session_state.counter += 1
st.write(f"Counter value: {st.session_state.counter}")

if st.button("Add st.query_params"):
    st.query_params["foo"] = "bar"
    st.query_params["timestamp"] = str(datetime.datetime.now())

st.write("Current query params:", dict(st.query_params))


"## Advanced Examples"

"### Real-time data simulation"
if st.button("Start real-time chart"):
    chart_placeholder = st.empty()
    
    for i in range(50):
        new_data = pd.DataFrame({
            'x': range(i),
            'y': np.random.randn(i).cumsum()
        })
        
        chart_placeholder.line_chart(new_data, x='x', y='y')
        time.sleep(0.1)

"### Interactive data filtering"
sample_data = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D', 'E'] * 20,
    'Sales': np.random.randint(100, 1000, 100),
    'Profit': np.random.randint(10, 100, 100),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100)
})

selected_region = st.multiselect(
    "Filter by region:",
    options=sample_data['Region'].unique(),
    default=sample_data['Region'].unique().tolist()
)

filtered_data = sample_data[sample_data['Region'].isin(selected_region)]

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${filtered_data['Sales'].sum():,}")
col2.metric("Avg Profit", f"${filtered_data['Profit'].mean():.2f}")
col3.metric("Records", len(filtered_data))

st.bar_chart(filtered_data.groupby('Product')['Sales'].sum())

"### Dynamic form builder"
num_fields = st.slider("Number of form fields", 1, 5, 3)

with st.form("dynamic_form"):
    st.write("Dynamic Form")
    inputs = {}
    for i in range(num_fields):
        inputs[f"field_{i}"] = st.text_input(f"Field {i+1}")
    
    submitted = st.form_submit_button("Submit All")
    if submitted:
        st.write("Form data:", inputs)

"### Multi-step process"
if "step" not in st.session_state:
    st.session_state.step = 1

st.write(f"### Step {st.session_state.step} of 3")

if st.session_state.step == 1:
    st.write("Enter your basic information")
    name = st.text_input("Name")
    if st.button("Next →"):
        st.session_state.step = 2
        st.rerun()
        
elif st.session_state.step == 2:
    st.write("Choose your preferences")
    preferences = st.multiselect("Select options", ["Option A", "Option B", "Option C"])
    col1, col2 = st.columns(2)
    if col1.button("← Back"):
        st.session_state.step = 1
        st.rerun()
    if col2.button("Next →"):
        st.session_state.step = 3
        st.rerun()
        
elif st.session_state.step == 3:
    st.write("Review and confirm")
    st.success("All steps completed!")
    col1, col2 = st.columns(2)
    if col1.button("← Back"):
        st.session_state.step = 2
        st.rerun()
    if col2.button("Reset"):
        st.session_state.step = 1
        st.rerun()

"### Color palette generator"
num_colors = st.slider("Number of colors", 3, 10, 5, key="color_palette")
cols = st.columns(num_colors)
colors = []

for i, col in enumerate(cols):
    color = col.color_picker(f"Color {i+1}", f"#{i*30:02x}{i*40:02x}{i*50:02x}")
    colors.append(color)
    col.markdown(f"<div style='background-color:{color};height:100px;'></div>", unsafe_allow_html=True)

st.code(f"colors = {colors}")

"### Tabs with different content types"
tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📝 Text", "🎨 Media", "🔢 Data"])

with tab1:
    chart_data = pd.DataFrame(
        np.random.randn(50, 3),
        columns=['Series A', 'Series B', 'Series C']
    )
    st.line_chart(chart_data)
    st.bar_chart(chart_data)

with tab2:
    st.markdown("### Rich Text Content")
    st.write("This tab contains various text elements:")
    st.info("Information box with important details")
    st.warning("Warning message for users")
    st.code("print('Hello, Streamlit!')", language="python")

with tab3:
    st.markdown("### Media Gallery")
    img_cols = st.columns(3)
    for i, col in enumerate(img_cols):
        col.image(f"https://picsum.photos/200/200?random={i+10}", use_container_width=True)

with tab4:
    st.markdown("### Data Tables")
    table_data = pd.DataFrame(
        np.random.randn(10, 5),
        columns=[f"Column {i+1}" for i in range(5)]
    )
    st.dataframe(table_data, use_container_width=True)

"### Nested containers and layouts"
with st.container(border=True):
    st.subheader("Main Container")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.container(border=True):
            st.write("Nested container 1")
            st.metric("Value", 100, 10)
    
    with col2:
        with st.container(border=True):
            st.write("Nested container 2")
            st.button("Action")

"### Conditional UI elements"
show_advanced = st.checkbox("Show advanced options", False)

if show_advanced:
    with st.expander("Advanced Settings", expanded=True):
        st.slider("Parameter 1", 0, 100, 50)
        st.slider("Parameter 2", 0, 100, 50)
        st.selectbox("Algorithm", ["Option A", "Option B", "Option C"])
        st.checkbox("Enable feature X")
        st.checkbox("Enable feature Y")

"### Progress tracking"
if st.button("Run multi-step process"):
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    steps = ["Initializing", "Loading data", "Processing", "Analyzing", "Finalizing"]
    
    for i, step in enumerate(steps):
        progress_text.write(f"**{step}...**")
        progress_bar.progress((i + 1) / len(steps))
        time.sleep(0.8)
    
    progress_text.write("✅ **Complete!**")
    st.balloons()

"### Data visualization comparison"
viz_type = st.radio(
    "Select visualization type:",
    ["Line", "Bar", "Area", "Scatter"],
    horizontal=True
)

sample_viz_data = pd.DataFrame({
    'x': range(20),
    'y1': np.random.randn(20).cumsum(),
    'y2': np.random.randn(20).cumsum(),
    'y3': np.random.randn(20).cumsum()
})

if viz_type == "Line":
    st.line_chart(sample_viz_data, x='x', y=['y1', 'y2', 'y3'])
elif viz_type == "Bar":
    st.bar_chart(sample_viz_data, x='x', y=['y1', 'y2', 'y3'])
elif viz_type == "Area":
    st.area_chart(sample_viz_data, x='x', y=['y1', 'y2', 'y3'])
elif viz_type == "Scatter":
    st.scatter_chart(sample_viz_data, x='x', y=['y1', 'y2', 'y3'])

"### Custom styled metrics dashboard"
st.markdown("### Performance Dashboard")
metric_cols = st.columns(4)

metrics = [
    ("Revenue", "$125K", "+12%", "🟢"),
    ("Users", "2,345", "+5%", "🟢"),
    ("Engagement", "67%", "-3%", "🔴"),
    ("Retention", "89%", "+8%", "🟢")
]

for col, (label, value, delta, icon) in zip(metric_cols, metrics):
    with col:
        st.metric(label, value, delta, border=True)
        st.markdown(f"<center>{icon}</center>", unsafe_allow_html=True)

"### Interactive calculator"
st.markdown("### Simple Calculator")
calc_col1, calc_col2, calc_col3 = st.columns(3)

num1 = calc_col1.number_input("First number", value=10.0)
operation = calc_col2.selectbox("Operation", ["+", "-", "×", "÷"])
num2 = calc_col3.number_input("Second number", value=5.0)

if operation == "+":
    result = num1 + num2
elif operation == "-":
    result = num1 - num2
elif operation == "×":
    result = num1 * num2
elif operation == "÷":
    result = num1 / num2 if num2 != 0 else "Error: Division by zero"

st.markdown(f"### Result: `{result}`")

"### Collapsible sections with state"
sections = ["Section A", "Section B", "Section C", "Section D"]

for section in sections:
    with st.expander(f"📁 {section}"):
        st.write(f"Content for {section}")
        st.slider(f"Slider for {section}", 0, 100, 50, key=f"slider_{section}")
        st.text_input(f"Input for {section}", key=f"input_{section}")

"### File download examples"
csv_data = pd.DataFrame({
    'Column1': [1, 2, 3, 4, 5],
    'Column2': ['A', 'B', 'C', 'D', 'E'],
    'Column3': [10.5, 20.3, 30.7, 40.1, 50.9]
})

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        label="Download CSV",
        data=csv_data.to_csv(index=False),
        file_name='data.csv',
        mime='text/csv',
    )

with col2:
    st.download_button(
        label="Download JSON",
        data=csv_data.to_json(orient='records'),
        file_name='data.json',
        mime='application/json',
    )

with col3:
    st.download_button(
        label="Download Text",
        data="This is sample text content\nLine 2\nLine 3",
        file_name='sample.txt',
        mime='text/plain',
    )

"### End of demo"
st.divider()
st.caption("This demo showcases various Streamlit components with Anthropic dark theme styling")
st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
