"""Generate narration audio for every line in video/script.json.

Uses qwen-audio-3.1-tts-next: `text` = what is spoken,
`text_prompt` = how to say it (voice style + background music).
A per-line duration manifest is written to video/timings.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import synth, load_key  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "video", "script.json")

BASE = (
    "成熟的中文男声解说，发音标准清晰，语速中等偏快，声音有磁性，"
    "叙事感强，像在讲一段自己亲身经历的技术故事。"
)

MOOD = {
    "hook": "语气笃定而克制，第一句要有分量，结尾带一点压抑后的释放感。"
            "背景音乐：史诗感弦乐缓缓推进，鼓点进入，音量低不盖人声。",
    "ram_fail": "语气平静略带自嘲，像复盘一次失败。"
                "背景音乐：低沉的电子脉冲，轻微悬疑感。",
    "cnn": "语气转为明朗，讲清楚思路的转变。"
           "背景音乐：加入轻快的合成器琶音，节奏稳定。",
    "label": "语气无奈但幽默，列举翻车经历。"
             "背景音乐：俏皮的低音拨弦，轻松但不喧闹。",
    "tanh": "语气压低、强调严重性，讲到数字时加重。"
            "背景音乐：低频轰鸣渐强，制造危机感。",
    "perf": "语气轻快，讲提速时带一点得意。"
            "背景音乐：节奏感强的电子鼓，能量上扬。",
    "lie": "语气先松后紧，转折处停顿一下，制造反转。"
           "背景音乐：短暂安静后低音骤起。",
    "plateau": "语气先沉后扬，最后一句要有冲破感。"
               "背景音乐：持续铺底后逐渐推向高潮。",
    "clear": "情绪明显上扬，爽快、有成就感。"
             "背景音乐：铜管与鼓齐奏，胜利感。",
    "challenge": "语气坚定有力，像立下承诺，最后一句放慢收住。"
                 "背景音乐：宏大开阔的管弦乐，收尾留白。",
}


def main():
    lines = json.load(open(SCRIPT, encoding="utf-8"))
    key = load_key()
    timings = []
    for item in lines:
        print("->", item["id"])
        prompt = BASE + MOOD.get(item.get("kind", ""), "")
        path = synth(item["text"], prompt, item["name"], key)
        timings.append({
            "id": item["id"],
            "name": item["name"],
            "audio": os.path.relpath(path, ROOT).replace("\\", "/"),
            "title": item["title"],
            "sub": item["sub"],
            "kind": item["kind"],
            "visual": item["visual"],
            "text": item["text"],
        })
    with open(os.path.join(ROOT, "video", "timings_raw.json"), "w",
              encoding="utf-8") as f:
        json.dump(timings, f, ensure_ascii=False, indent=2)
    print("wrote timings_raw.json")


if __name__ == "__main__":
    main()
