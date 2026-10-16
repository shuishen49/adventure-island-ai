"""Deep diagnosis of qwen-audio-3.1-tts-next behaviour.

Checks: (1) proof of which model answered, (2) how duration scales with
text length, (3) whether speaking rate can be controlled.
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import load_key  # noqa: E402

END = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
MODEL = "qwen-audio-3.1-tts-next"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "tts_diag")
os.makedirs(OUT, exist_ok=True)


def call(text, prompt, tag, extra=None):
    body = {"model": MODEL, "input": {"text": text, "text_prompt": prompt}}
    if extra:
        body["input"].update(extra)
    req = urllib.request.Request(
        END, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + load_key(),
                 "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    a = r["output"]["audio"]
    path = os.path.join(OUT, tag + ".wav")
    urllib.request.urlretrieve(a["url"], path)
    chars = len(text.replace("，", "").replace("。", ""))
    d = a.get("duration", 0)
    rate = chars / d if d > 0 else 0
    print(f"{tag:16s} chars={chars:3d} dur={d:6.2f}s  rate={rate:5.2f} 字/秒  "
          f"(model={r['output'].get('model', '?') if 'model' in r['output'] else 'n/a'})")
    return r, path


if __name__ == "__main__":
    # 1) proof: dump the raw response for one call
    r, _ = call("测试。", "中文男声，自然朗读。", "proof")
    print("\n--- raw response ---")
    print(json.dumps(r, ensure_ascii=False)[:700])

    print("\n--- duration vs text length (pure speech, no BGM) ---")
    base = "成熟的中文男声，自然朗读，语速正常。"
    texts = {
        "len_short": "这一次，WorkBuddy 立了大功。",
        "len_mid": "最开始我想得很简单：把内存里的玩家坐标、敌人位置、速度抓出来喂给 PPO，不就行了？",
        "len_long": "这一次，WorkBuddy 立了大功。我折腾了好几个月的《冒险岛》，这次三天，就训练通关了。",
    }
    for tag, t in texts.items():
        call(t, base, tag)

    print("\n--- explicit rate control on the long line ---")
    long = texts["len_long"]
    for tag, p in {
        "rate_slow": base + "语速放慢一些，每分钟约 200 字，句间有自然停顿。",
        "rate_normal": base + "语速每分钟约 260 字，句子之间有停顿。",
        "rate_narration": "中文男声纪录片旁白，语速舒缓沉稳，每分钟约 220 字。",
    }.items():
        call(long, p, tag)
