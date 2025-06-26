# parse_txt.py
import re, pathlib, json

text = pathlib.Path("HandWrittenQA.txt").read_text()

pattern = re.compile(
    r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)",
    re.S                                    
)

qa_pairs = [{"q": q.strip(), "a": a.strip()} for q, a in pattern.findall(text)]

with open("qa.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print(f"Extracted {len(qa_pairs)} Q-A pairs → qa.json")
