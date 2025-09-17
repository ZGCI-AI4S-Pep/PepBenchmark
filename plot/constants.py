from pathlib import Path
from matplotlib import font_manager as fm


data_dir = Path("/Users/zhoukuangqi/Desktop/pepbenchmark/dataset_statistic")
meta_data_dir = Path("/Users/zhoukuangqi/Desktop/pepbenchmark/plot/meta_data")
output_dir = Path("/Users/zhoukuangqi/Desktop/pepbenchmark/plot/outputs")

font_path = (
    "/Users/zhoukuangqi/Desktop/pepbenchmark/plot/Fira_Sans/FiraSans-Regular.ttf"
)
font_prop = fm.FontProperties(fname=font_path)

blues = [
    "#cfdaea",
    "#c0d4f5",
    "#afcafc",
    "#9ebeff",
    "#8db0fe",
    "#7b9ff9",
    "#6a8bef",
    "#5977e3",
    "#4961d2",
    "#3b4cc0",
]
reds = [
    "#dddcdc",
    "#e9d5cb",
    "#f2cbb7",
    "#f6bda2",
    "#f7ac8e",
    "#f4987a",
    "#ee8468",
    "#e36c55",
    "#d65244",
    "#c53334",
]
