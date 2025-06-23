project = 'Python for ML algorithms'
author = 'Brian Staber'
release = '0.1'

extensions = [
    "myst_parser",
    "sphinx_inline_tabs",
    "sphinx_copybutton"
]

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

source_suffix = {
    '.md': 'markdown',
}

html_theme = 'furo'
html_title = "Python for ML algorithms"

copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False