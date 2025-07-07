# Copyright ZGCA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import sys

# docs/source/conf.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project-root
SRC = ROOT / "src"  # project-root/src
sys.path.insert(0, SRC.as_posix())

project = "pepbenchmark"
copyright = "2025, ZGCA"
author = "ZGCA"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 官方扩展
    "sphinx.ext.autodoc",  # 自动从 docstring 生成文档
    "sphinx.ext.autosummary",  # 自动生成汇总表
    "sphinx.ext.napoleon",  # 支持 NumPy / Google 风格 docstring
    "sphinx.ext.intersphinx",  # 链接到外部文档
    "sphinx.ext.todo",  # TODO 列表
    "sphinx.ext.viewcode",  # 在文档里显示源码链接
    # 社区扩展
    "myst_parser",  # 让 Sphinx 解析 Markdown (*.md)
]
autosummary_generate = True  # 生成 autosummary stub 文件
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
html_theme = "sphinx_rtd_theme"

templates_path = ["_templates"]
exclude_patterns = []
version = "1.0"
release = "1.0"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
