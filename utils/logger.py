from datetime import datetime

def print_log(tag:str,content:str):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}]【{tag}】{content}")