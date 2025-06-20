from IPython.display import HTML

def display_scroll(df, height=300):
    style = f"height:{height}px; overflow:auto; border:1px solid lightgray;"
    html = df.to_html()
    return HTML(f"<div style='{style}'>{html}</div>")
