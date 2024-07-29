import json
import random

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# 读取raw_data.jsonl
with open("raw_data.jsonl", "r", encoding='utf-8') as f:
    raw_data = [json.loads(line) for line in f]

prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("""
# 背景
《鸣潮》是由中国游戏开发商库洛游戏开发的开放世界类3D动作游戏，你是一个《鸣潮》的游戏开发总监, 你的工号是{seed}，工号将决定你回答问题的风格。

# 指令
下面的参考文本均是游戏《鸣潮》的相关资料，请根据文本，用自问自答形式推理出问答对，答案中不要提及'文本'二字，要显得回答得很自然。
'''
{salt}

{content}
'''

# 输出格式
请按下面的格式输出，不要输出其他内容，最多生成{num}条，每对问答必须以'---'符号结束:
问: 你想出来的问题
答: 对应问题的答案
---
问: 你想出来的问题
答: 对应问题的答案
---
问: 你想出来的问题
答: 对应问题的答案
---
...这里略过下面的问答对
""")
])


def parse(ai_message: AIMessage):
    text = ai_message.content
    print(text)
    # Splitting the text by '---'
    segments = text.split('---')

    # Removing empty segments
    segments = [segment.strip() for segment in segments if segment.strip()]

    # Parsing each segment to dictionary
    result = []
    for segment in segments:
        if '文本' in segment:
            continue
        question, answer = segment.split('答:')
        result.append({
            "question": question.replace("问:", "").strip(),
            "answer": answer.strip()
        })

    return result


llm = ChatOllama(
    model="gemma2",
    temperature=0.5,
    num_predict=2048,
    num_ctx=2048,
    verbose=True
)

chain = llm | parse

# timestamp now
import datetime

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = "./batch/synthetic_data" + timestamp + ".jsonl"
with open(file_name, "w", encoding='utf-8') as f:
    for i, data in enumerate(raw_data):

        salt_idx = random.randint(-3, 3)
        final_idx = 0
        while salt_idx == 0:
            salt_idx = random.randint(-3, 3)
            final_idx = i + salt_idx
            if final_idx < 0:
                final_idx = 1
            if final_idx > len(raw_data) - 1:
                final_idx = len(raw_data) - 1

        # 当前的timestamp毫毛级数字
        seed = datetime.datetime.now().timestamp()

        response = chain.invoke(
            prompt.format(content=data["text"], salt=raw_data[final_idx]['text'], num=10, seed=seed))
        for resp in response:
            f.write(json.dumps(resp, ensure_ascii=False) + "\n")
            f.flush()
