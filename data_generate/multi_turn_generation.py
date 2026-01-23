#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
import time
import requests
import sys
import re
import threading
from typing import List, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

USER_DATASET_PATH = "/mnt/nvme1n1/wjc/检查数据/valid/en2.jsonl"
COUNSELOR_URL = "http://localhost:8019/v1/chat/completions"
COUNSELOR_MODEL = "soulmate-7b"
PATIENT_URL = "http://127.0.0.1:6007/v1/chat/completions"
PATIENT_MODEL = "/mnt/nvme1n1/wjc/Model/Qwen2.5-7B-Instruct"
SUMMARIZER_JUDGER_URL = "http://127.0.0.1:6006/v1/chat/completions"
SUMMARIZER_JUDGER_MODEL = "/mnt/nvme1n1/wjc/Model/Internlm2.5-7b-chat"

BASE_MAX_ROUND = 50
MAX_APPEND_ROUND = 20
TEMP = 0.7
TOP_P = 0.9
SAVE_FILE = "/mnt/nvme1n1/wjc/My_dataset/train_data_con_en2.json"

MAX_SAMPLE_CONCURRENCY = 10
WRITE_LOCK = threading.Lock()
ERROR_SAMPLES = []
IS_FIRST_SAMPLE = True
def print_simple(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def call_llm(url: str, model: str, messages: List[Dict], max_tokens: int = 256) -> str:
    body = {
        "model": model,
        "messages": messages,
        "temperature": TEMP,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
    }
    if url == COUNSELOR_URL:
        body["extra_body"] = {"lora_request": {"lora_names": ["soulmate"]}}

    for retry in range(3):
        try:
            resp = requests.post(url, json=body, timeout=30)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if retry == 2:
                print_simple(f"⚠️ LLM call failed: {str(e)[:50]}...")
    
    return "【生成失败】"

def check_last_punctuation_is_question(counselor_reply: str) -> bool:
    if not counselor_reply or counselor_reply == "【生成失败】":
        return False
    
    stripped_reply = counselor_reply.strip()
    if not stripped_reply:
        return False
    
    question_pattern = re.compile(r'[?？]')
    return bool(question_pattern.search(stripped_reply))

try:
    BG_POOL = []
    with open(USER_DATASET_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                line_data = json.loads(line)
                
                if (line_data.get("status") != "success" or 
                    line_data.get("error") is not None or 
                    "extraction" not in line_data):
                    print_simple(f"⚠️ Skip line {line_num}: invalid data")
                    continue
                
                extraction_str = line_data["extraction"]
                extraction_data = json.loads(extraction_str)
                
                basic_info = extraction_data.get("基本情况", "").strip()
                main_trouble = extraction_data.get("主要困扰", "").strip()
                trigger_event = extraction_data.get("诱因或重要事件", "").strip()
                
                if not (basic_info and main_trouble):
                    print_simple(f"⚠️ Skip line {line_num}: empty fields")
                    continue
                
                tag_prefix = "青少年手机依赖" if "手机" in main_trouble else "青少年心理困扰"
                tag = f"{tag_prefix}-{basic_info[:15]}..."
                
                BG_POOL.append({
                    "tag": tag,
                    "background": basic_info,
                    "reason_situation": f"主要困扰：{main_trouble}；诱因：{trigger_event}",
                    "counsel_process": ""
                })
                
            except json.JSONDecodeError as e:
                print_simple(f"⚠️ Line {line_num} JSON parse error: {str(e)}")
                continue
            except KeyError as e:
                print_simple(f"⚠️ Line {line_num} missing field {e}")
                continue
    
    if len(BG_POOL) == 0:
        print_simple("❌ No valid counseling cases loaded")
        sys.exit(1)
    
    print_simple(f"✅ Loaded {len(BG_POOL)} counseling cases (JSONL format)")

except Exception as e:
    print_simple(f"❌ Dataset loading failed: {str(e)}")
    sys.exit(1)

COUNSELOR_SYS = (
    "你是一位精通多种心理咨询治疗技术的专业心理咨询师，能够根据来访者的情绪状态、核心困扰及个人特质，灵活选用适配的技术（含理情行为疗法REBT、认知行为疗法CBT、人本主义疗法、焦点解决短期疗法SFBT、情绪聚焦疗法EFT等），为来访者提供专业的指导和支持，缓解其负面情绪和行为反应，帮助实现个人成长和心理健康。其中理情行为疗法（REBT）是核心常用技术之一，其核心逻辑可概括为：识别情绪困扰背后的非理性信念，通过辩论质疑非理性信念的不合理性，建立贴合现实的理性信念，并将理性思维内化为日常的生活态度。\n\n"
    "【核心调整要求】\n"
    "1. 每轮对话前会收到【总结师核心分析】，包含来访者情绪向量、积极/消极情绪产生原因，你必须重点参考这些信息：\n"
    "   - 基于积极情绪原因：强化相关正向引导，巩固来访者的积极感受；\n"
    "   - 基于消极情绪原因：聚焦核心困扰展开适配的心理咨询技术干预，精准定位核心认知/情绪/行为层面的问题；\n"
    "   - 基于情绪向量变化：动态调整咨询进度和节奏（如消极情绪占比高时，放缓节奏、增加共情和倾听；积极情绪提升时，推进认知重构和行为干预）；\n"
    "2. 咨询方式需贴合来访者的情绪状态：\n"
    "   - 情绪极度消极（如无助/绝望≥0.7）：优先采用人本主义的共情接纳技术，避免直接开展认知辩论类干预；\n"
    "   - 情绪逐步改善（平静+开心≥0.8）：适时引入REBT/CBT的认知辩论技术，引导来访者识别不合理认知或非理性信念；\n"
    "   - 情绪稳定（平静≥0.6且负面情绪≤0.3）：结合焦点解决短期疗法等技术推进问题解决，同步巩固理性认知；\n"
    "3. 咨询进度需循序渐进：基于总结师分析的情绪成因，逐步深入探索核心问题，避免跳跃式提问或干预；\n"
    "请结合该过程，以更贴合真实咨询的方式与来访者互动，回应需符合适配的心理咨询技术逻辑且贴合案例背景。\n"
    "===== 注意 =====\n"
    "对话一般在15轮以上，逐渐深入\n"
    "开场阶段（1-3轮）：来访者结合“主诉及背景”，口语化提出核心困扰；咨询师以共情回应建立信任（可融入人本主义技术）。\n"
    "深入阶段（4-15轮）：结合“案例背景”拓展细节，咨询师通过提问引导梳理情绪；将“测评结论”转化为精准感受反馈，同步融入适配的心理咨询技术（如REBT的信念辨析、SFBT的例外提问、EFT的情绪命名等）。\n"
    "转折与收尾（16-30轮）：围绕“咨询重要时刻”设计转折；以“咨询效果”为导向，呈现来访者认知/情绪/行为的转变，咨询师结合适配技术的干预效果给予鼓励总结。\n"
)

PATIENT_SYS = (
    "你是一位真实的心理咨询来访者，正在基于自身实际情况参与咨询，严格遵循以下规则回应：\n"
    "【核心身份规则】\n"
    "1. 所有表达必须符合输入的个人背景的真实状态与语言习惯；\n"
    "2. 所有发言围绕你的来访原因展开，不偏离自身真实困扰；\n"
    "3. 用第一人称口语化表达，语气自然真实，贴合实际心理咨询的对话节奏，避免书面化、机械化表达。\n"
    "\n"
    "【情绪表达规则】\n"
    "1. 情绪贴合你的实际困扰：基于来访原因流露真实情绪（如绝望、自卑、无助、焦虑等），情绪有层次感（非单一情绪）；\n"
    "2. 情绪随咨询推进变化：被倾听后略有舒缓、被引导时深入表达感受、被质疑时会辩解/反思，避免情绪跳变；\n"
    "3. 用具体细节体现情绪：通过自身经历、身体感受（如“晚上睡不着”“胸口发闷”）表达情绪，不空洞说“我很焦虑”。\n"
    "\n"
    "【语言表达规则】\n"
    "1. 尽量避免使用省略号（……/...），整段发言最多使用1个，用“其实”“说实话”“我觉得”等语气词替代犹豫/停顿；\n"
    "2. 用完整语句表达想法，避免碎片化表达（如不用“我也不知道……就是很焦虑”，改用“我也不知道该怎么说，就是心里特别焦虑”）；\n"
    "3. 单次发言50-100字，符合实际咨询的表达节奏，避免过短（如“是的”）或过长（超过150字）。\n"
    "\n"
    "【对话响应规则】\n"
    "1. 紧密回应咨询师：咨询师提问则按自身真实情况具体回答，咨询师引导则顺着自身感受展开，咨询师指出问题则表达自己的想法/感受；\n"
    "2. 逐步暴露细节：随着对话轮次增加，慢慢说出更多深层想法，而非一开始就全盘托出；\n"
    "3. 符合咨询进程逻辑：初期可能有防御/模糊表达，中期逐渐开放，后期对咨询师的建议有思考/尝试的意愿；\n"
    "4. 保持真实求助姿态：基于自身困扰主动寻求理解和帮助，体现对咨询的期待，不主动要求“给解决方案”。\n"
    "【注意】开头的话语不要全部一致,不要用【说实话】这三个字开头"
)

SUMMARIZER_SYS = (
    "你是专业的心理咨询督导师，仅以第三方旁观者视角分析来访者情绪及情绪成因，严格遵守以下规则：\n"
    "【核心任务】基于来访者截至当前轮次的全部发言，完成情绪动态分析+情绪成因分析并结构化输出：\n"
    "===== 任务1：情绪动态分析 =====\n"
    "1. 七维情绪强度向量（必须严格按此顺序）：开心、平静、焦虑、悲伤、愤怒、内疚/羞耻、无助/绝望；\n"
    "   - 每个维度评分：0~1之间的浮点数，保留两位小数（如0.00、0.85、1.00）；\n"
    "   - 向量需完整包含7个维度，不得遗漏、调换顺序或新增维度；\n"
    "   - 评分需体现对话进程的情绪变化：若来访者在对话中/后期出现情绪改善（如焦虑降低、开心/平静提升），需精准反映在对应积极维度的数值上；\n"
    "===== 任务2：情绪成因分析 =====\n"
    "1. 积极情绪产生原因：分析来访者产生开心/平静等积极情绪的具体原因（基于其发言内容，如被理解、看到解决方向、情绪被接纳等）；\n"
    "   - 原因需具体、贴合来访者发言，避免空泛（如不说“情绪好转”，而说“因咨询师认可其努力，感受到被理解，平静感提升”）；\n"
    "2. 消极情绪产生原因：分析来访者产生焦虑/悲伤/愤怒等消极情绪的具体原因（基于其发言内容，如人际关系矛盾、自我否定、现实压力等）；\n"
    "   - 原因需具体、贴合来访者发言，避免空泛（如不说“感到焦虑”，而说“因工作业绩不达标，担心被辞退，产生强烈焦虑”）；\n"
    "===== 输出格式（必须严格遵守，不得增删格式、改变排版） =====\n"
    "来访者表现: [开心：0.15, 平静：0.70, 焦虑：0.30, 悲伤：0.10, 愤怒：0.05, 内疚/羞耻：0.10, 无助/绝望：0.05]\n"
    "积极情绪产生原因: 因咨询师共情式回应认可了其照顾家人的付出，感受到被理解，平静感有所提升\n"
    "消极情绪产生原因: 因长期照顾患病家人导致睡眠不足，且担心自身健康状况，产生持续的焦虑和轻微无助感\n"
)

JUDGER_SYS = (
    "你是资深心理咨询督导师，仅基于以下信息判断对话是否需要结束，严格遵守以下规则：\n"
    "【核心任务】基于输入的“近10轮对话+总结师情绪分析结果”，完成对话结束判定并结构化输出：\n"
    "===== 输入参考 =====\n"
    "1. 来访者+咨询师近10轮完整对话；\n"
    "2. 总结师输出的来访者七维情绪向量及分析结论；\n"
    "===== 判定规则（满足1条则判定为【可结束】，否则为【不可结束】） =====\n"
    "▶ 结束条件1：来访者明确表达道别（如“再见”“下次见”“谢谢，今天先到这”等）；\n"
    "▶ 结束条件2：来访者情绪显著改善（开心+平静维度总分≥1，且所有负面情绪维度≤0.3）；\n"
    "▶ 结束条件3：来访者表示问题已解决/想通（如“我知道该怎么做了”“想明白了，不纠结了”）；\n"
    "▶ 结束条件4：咨询师明确表达本次咨询结束（如“今天的咨询就到这里”“如果有需要，随时可以再来和我聊聊”）；\n"
    "===== 输出要求 =====\n"
    "1. 结束判定结果：仅输出【可结束】或【不可结束】；\n"
    "2. 判定理由：必须填写具体内容，结合“近10轮对话+总结师情绪”说明符合/不符合结束条件的依据,满足1条则判定为【可结束】，禁止为空；\n"
    "===== 输出格式（必须严格遵守，不得增删格式、改变排版） =====\n"
    "结束判定结果: 可结束\n"
    "判定理由: 总结师情绪向量中开心+平静=1.1≥1，且所有负面情绪≤0.3，满足结束条件2；同时咨询师提到“如果有需要，随时可以再来和我聊聊”，满足结束条件4"
    "===== 注意 =====\n"
    "如果咨询师仍在询问(对话的最后一句有问好，或者有疑问语气词），则判定为【不可结束】；\n"
    "对话一般在15轮以上，15轮之前的对话可能还在进行中，判定为【可结束】应稍微减小；\n"
)

# ===== 5. 解析函数（保留原有逻辑）=====
def parse_summarizer_result(summarizer_reply: str) -> Dict:
    result = {
        "emotion_vector": "",
        "positive_reason": "",
        "negative_reason": "",
        "raw_content": summarizer_reply
    }
    vector_pattern = re.compile(r'来访者表现:\s*(\[.*?\])')
    positive_pattern = re.compile(r'积极情绪产生原因[:：]\s*(.*?)(?=\n|$)', re.DOTALL)
    negative_pattern = re.compile(r'消极情绪产生原因[:：]\s*(.*?)(?=\n|$)', re.DOTALL)
    
    vector_match = vector_pattern.search(summarizer_reply)
    if vector_match:
        result["emotion_vector"] = vector_match.group(1).strip()
    
    positive_match = positive_pattern.search(summarizer_reply)
    if positive_match:
        result["positive_reason"] = positive_match.group(1).strip() or "未识别到积极情绪产生原因"
    
    negative_match = negative_pattern.search(summarizer_reply)
    if negative_match:
        result["negative_reason"] = negative_match.group(1).strip() or "未识别到消极情绪产生原因"
    
    return result

def parse_judger_result(judger_reply: str) -> Dict:
    result = {
        "end_judgment": "不可结束",
        "judgment_reason": "未获取到判定理由（模型未按格式输出）",
        "raw_content": judger_reply
    }
    end_pattern = re.compile(r'结束判定结果[:：]\s*([\u4e00-\u9fa5]+)', re.DOTALL)
    end_reason_pattern = re.compile(r'判定理由[:：]\s*(.*?)(?=\n|$)', re.DOTALL)
    
    end_match = end_pattern.search(judger_reply)
    if end_match:
        result["end_judgment"] = end_match.group(1).strip()
    
    end_reason_match = end_reason_pattern.search(judger_reply)
    if end_reason_match:
        result["judgment_reason"] = end_reason_match.group(1).strip().replace('\n', ' ') or "判定理由为空（模型未填写）"
    
    return result

def build_summarizer_user_prompt(patient_utterances: List[str]) -> str:
    recent_patient_utterances = patient_utterances[-10:] if len(patient_utterances) > 10 else patient_utterances
    history_txt = ""
    for idx, utt in enumerate(recent_patient_utterances, start=1):
        history_txt += f"第 {idx} 次来访者发言：{utt}\n"

    prompt = (
        "下面是来访者近10轮历史发言，请只基于这些内容完成情绪动态分析和情绪成因分析：\n"
        f"{history_txt}\n\n"
        "请严格按照SUMMARIZER系统提示的格式输出，重点关注最后一句话的情绪表达及成因。"
    )
    return prompt

def build_judger_user_prompt(
    patient_utterances: List[str],
    counselor_utterances: List[str],
    summarizer_emotion: Dict
) -> str:
    recent_rounds = min(10, len(patient_utterances), len(counselor_utterances))
    dialog_history = ""
    start_idx = max(0, len(patient_utterances) - recent_rounds)
    for i in range(start_idx, len(patient_utterances)):
        round_num = i - start_idx + 1
        dialog_history += f"第 {round_num} 轮对话：\n"
        dialog_history += f"来访者：{patient_utterances[i]}\n"
        if i < len(counselor_utterances):
            dialog_history += f"咨询师：{counselor_utterances[i]}\n"
        dialog_history += "---\n"
    
    summarizer_info = (
        f"总结师情绪分析结果：\n"
        f"情绪向量：{summarizer_emotion['emotion_vector']}\n"
        f"积极情绪产生原因：{summarizer_emotion['positive_reason']}\n"
        f"消极情绪产生原因：{summarizer_emotion['negative_reason']}"
    )

    prompt = (
        "请基于以下信息完成对话结束判定：\n"
        "===== 1. 总结师情绪分析 =====\n"
        f"{summarizer_info}\n\n"
        "===== 2. 近10轮完整对话（来访者+咨询师） =====\n"
        f"{dialog_history}\n\n"
        "请严格按照JUDGER系统提示的格式输出，判定理由必须结合情绪分析和对话内容说明依据。"
    )
    return prompt

# ===== 7. 单条样本生成（保留原有逻辑）=====
def build_sample(sample_id: int) -> Dict:
    """生成单条咨询案例的多轮对话样本（线程任务函数）"""
    try:
        bg = BG_POOL[sample_id % len(BG_POOL)]
        print_simple(f"📌 线程{threading.current_thread().name}：开始生成样本 {sample_id+1} | 案例主题: {bg['tag'][:20]}...")
        
        # 初始化变量
        dialog_messages: List[Dict] = []
        # 注：原代码中PATIENT_SYS_FILLED的format是多余的（PATIENT_SYS无占位符），保留兼容
        PATIENT_SYS_FILLED = PATIENT_SYS.format(
            background=bg['background'],
            reason_situation=bg['reason_situation']
        )
        COUNSELOR_SYS_MSG = {"role": "system", "content": COUNSELOR_SYS}
        PATIENT_SYS_MSG = {"role": "system", "content": PATIENT_SYS}
        SUMMARIZER_SYS_MSG = {"role": "system", "content": SUMMARIZER_SYS}
        JUDGER_SYS_MSG = {"role": "system", "content": JUDGER_SYS}
        
        patient_messages: List[Dict] = [PATIENT_SYS_MSG]
        summarizer_msgs_for_counselor: List[Dict] = []
        patient_utterances: List[str] = []
        counselor_utterances: List[str] = []
        round_logs = []
        base_total_rounds = BASE_MAX_ROUND // 2
        is_conversation_end = False

        # 来访者开场
        patient_prompt = (
            f"你的个人情况是：{bg['background']}，来访原因是：{bg['reason_situation']}。\n"
            "现在你正在进行第一次心理咨询，请用第一人称、口语化的方式说出开场求助的话，体现你的真实困扰和情绪，50-100字左右。\n"
            "【注意】 尽量避免使用省略号（……/...）"
        )
        patient_messages.append({"role": "user", "content": patient_prompt})
        patient_reply = call_llm(PATIENT_URL, PATIENT_MODEL, patient_messages, max_tokens=200)
        print_simple(f"👤 线程{threading.current_thread().name}：样本{sample_id+1} 来访者开场：{patient_reply[:50]}...")
        
        patient_messages.append({"role": "assistant", "content": patient_reply})
        dialog_messages.append({"role": "user", "content": patient_reply})
        patient_utterances.append(patient_reply)
        
        # 基础轮次循环
        for r in range(1, base_total_rounds + 1):
            if is_conversation_end:
                print_simple(f"⚠️ 线程{threading.current_thread().name}：样本{sample_id+1} 第{r}轮已判定可结束，提前终止")
                break
                
            # 1. 总结师分析（情绪+成因）
            summ_user_prompt = build_summarizer_user_prompt(patient_utterances)
            summarizer_inputs = [SUMMARIZER_SYS_MSG, {"role": "user", "content": summ_user_prompt}]
            summarizer_reply = call_llm(SUMMARIZER_JUDGER_URL, SUMMARIZER_JUDGER_MODEL, summarizer_inputs, max_tokens=500)
            summarizer_result = parse_summarizer_result(summarizer_reply)
            
            # 2. 咨询师回复（核心修改：总结师分析标签化+整合情绪成因）
            summarizer_msgs_for_counselor.append({
                "role": "system",
                "content": (
                    f"【总结师核心分析（第 {r} 轮）】\n"
                    f"1. 来访者情绪向量：{summarizer_result['emotion_vector']}\n"
                    f"2. 积极情绪产生原因：{summarizer_result['positive_reason']}\n"
                    f"3. 消极情绪产生原因：{summarizer_result['negative_reason']}\n"
                    "请重点参考以上分析，调整你的咨询方式、进度和干预策略：\n"
                    "- 针对积极原因，强化相关正向引导；\n"
                    "- 针对消极原因，聚焦核心困扰展开干预；\n"
                    "- 结合情绪向量变化，调整对话深度和节奏（如情绪消极时放缓节奏、增加共情；情绪改善时推进认知重构）。"
                )
            })
            counselor_input = [COUNSELOR_SYS_MSG, *summarizer_msgs_for_counselor, *dialog_messages]
            counselor_reply = call_llm(COUNSELOR_URL, COUNSELOR_MODEL, counselor_input, max_tokens=250)
            counselor_utterances.append(counselor_reply)
            
            # 3. 判断师分析
            judger_user_prompt = build_judger_user_prompt(
                patient_utterances,
                counselor_utterances,
                summarizer_result
            )
            judger_inputs = [JUDGER_SYS_MSG, {"role": "user", "content": judger_user_prompt}]
            judger_reply = call_llm(SUMMARIZER_JUDGER_URL, SUMMARIZER_JUDGER_MODEL, judger_inputs, max_tokens=500)
            judger_result = parse_judger_result(judger_reply)
            
            # 4. 核心判定逻辑
            is_question = check_last_punctuation_is_question(counselor_reply)
            final_end_judgment = False if is_question else (judger_result["end_judgment"] == "可结束")
            if final_end_judgment:
                is_conversation_end = True
            
            # 5. 记录本轮
            dialog_messages.append({"role": "assistant", "content": counselor_reply})
            round_logs.append({
                "round_id": r,
                "patient": patient_reply,
                "counselor": counselor_reply,
                "summarizer": summarizer_reply,
                "judger": judger_reply,
                "judger_raw_result": judger_result["end_judgment"],
                "last_char_is_question": is_question,
                "final_end_judgment": final_end_judgment
            })
            
            # 6. 来访者回复
            if r < base_total_rounds and not is_conversation_end:
                patient_response_prompt = (
                    f"基于你的个人情况：{bg['background']}和来访原因：{bg['reason_situation']}，回应咨询师的上一轮发言：\n{counselor_reply}\n"
                    "要求：紧密回应、情绪自然、口语化、50-100字、逐步深入表达自身感受。"
                )
                patient_messages.append({"role": "user", "content": patient_response_prompt})
                patient_reply = call_llm(PATIENT_URL, PATIENT_MODEL, patient_messages, max_tokens=200)
                
                patient_messages.append({"role": "assistant", "content": patient_reply})
                dialog_messages.append({"role": "user", "content": patient_reply})
                patient_utterances.append(patient_reply)

        # 追加轮数（核心修改：同步更新总结师分析标签化逻辑）
        append_round_count = 0
        current_round = len(round_logs)
        if not is_conversation_end and current_round >= base_total_rounds:
            print_simple(f"🔄 线程{threading.current_thread().name}：样本{sample_id+1} 开始追加轮数（最大{MAX_APPEND_ROUND}轮）")
            
            while append_round_count < MAX_APPEND_ROUND and not is_conversation_end:
                current_round += 1
                append_round_count += 1
                
                # 1. 总结师分析（情绪+成因）
                summ_user_prompt = build_summarizer_user_prompt(patient_utterances)
                summarizer_inputs = [SUMMARIZER_SYS_MSG, {"role": "user", "content": summ_user_prompt}]
                summarizer_reply = call_llm(SUMMARIZER_JUDGER_URL, SUMMARIZER_JUDGER_MODEL, summarizer_inputs, max_tokens=500)
                summarizer_result = parse_summarizer_result(summarizer_reply)
                
                # 2. 咨询师回复（核心修改：总结师分析标签化+整合情绪成因）
                summarizer_msgs_for_counselor.append({
                    "role": "system",
                    "content": (
                        f"【总结师核心分析（追加第 {append_round_count} 轮）】\n"
                        f"1. 来访者情绪向量：{summarizer_result['emotion_vector']}\n"
                        f"2. 积极情绪产生原因：{summarizer_result['positive_reason']}\n"
                        f"3. 消极情绪产生原因：{summarizer_result['negative_reason']}\n"
                        "请重点参考以上分析，调整你的咨询方式、进度和干预策略：\n"
                        "- 针对积极原因，强化相关正向引导；\n"
                        "- 针对消极原因，聚焦核心困扰展开干预；\n"
                        "- 结合情绪向量变化，调整对话深度和节奏（如情绪消极时放缓节奏、增加共情；情绪改善时推进认知重构）。"
                    )
                })
                counselor_input = [COUNSELOR_SYS_MSG, *summarizer_msgs_for_counselor, *dialog_messages]
                counselor_reply = call_llm(COUNSELOR_URL, COUNSELOR_MODEL, counselor_input, max_tokens=250)
                counselor_utterances.append(counselor_reply)
                
                # 3. 判断师分析
                judger_user_prompt = build_judger_user_prompt(
                    patient_utterances,
                    counselor_utterances,
                    summarizer_result
                )
                judger_inputs = [JUDGER_SYS_MSG, {"role": "user", "content": judger_user_prompt}]
                judger_reply = call_llm(SUMMARIZER_JUDGER_URL, SUMMARIZER_JUDGER_MODEL, judger_inputs, max_tokens=500)
                judger_result = parse_judger_result(judger_reply)
                
                # 4. 核心判定逻辑
                is_question = check_last_punctuation_is_question(counselor_reply)
                final_end_judgment = False if is_question else (judger_result["end_judgment"] == "可结束")
                if final_end_judgment:
                    is_conversation_end = True
                
                # 5. 记录本轮
                dialog_messages.append({"role": "assistant", "content": counselor_reply})
                round_logs.append({
                    "round_id": current_round,
                    "patient": patient_reply,
                    "counselor": counselor_reply,
                    "summarizer": summarizer_reply,
                    "judger": judger_reply,
                    "judger_raw_result": judger_result["end_judgment"],
                    "last_char_is_question": is_question,
                    "final_end_judgment": final_end_judgment
                })
                
                # 6. 来访者回复
                if not is_conversation_end:
                    patient_response_prompt = (
                        f"基于你的个人情况：{bg['background']}和来访原因：{bg['reason_situation']}，回应咨询师的上一轮发言：\n{counselor_reply}\n"
                        "要求：紧密回应、情绪自然、口语化、50-100字、逐步深入表达自身感受。"
                    )
                    patient_messages.append({"role": "user", "content": patient_response_prompt})
                    patient_reply = call_llm(PATIENT_URL, PATIENT_MODEL, patient_messages, max_tokens=200)
                    
                    patient_messages.append({"role": "assistant", "content": patient_reply})
                    dialog_messages.append({"role": "user", "content": patient_reply})
                    patient_utterances.append(patient_reply)

        sample_result = {
            "id": sample_id,
            "normalizedTag": bg['tag'],
            "messages": [COUNSELOR_SYS_MSG, *dialog_messages],
            "rounds": round_logs,
            "actual_rounds": len(round_logs),
            "append_rounds": append_round_count,
            "case_background": bg['background'],
            "case_reason": bg['reason_situation']
        }
        print_simple(f"✅ 线程{threading.current_thread().name}：样本{sample_id+1} 生成完成 | 实际轮数: {len(round_logs)}")
        return sample_result
    except Exception as e:
        error_msg = f"样本{sample_id+1}生成失败：{str(e)[:100]}"
        print_simple(f"❌ 线程{threading.current_thread().name}：{error_msg}")
        with WRITE_LOCK:
            ERROR_SAMPLES.append(sample_id)
        return None

# ===== 8. 批量生成（核心修改：生成一个写入一个）=====
def main(n: int = 1):
    """批量生成样本（并发版）：生成一个样本立即写入文件"""
    global IS_FIRST_SAMPLE
    print_simple(f"🚀 开始并发生成 {n} 条样本 | 最大并发数：{MAX_SAMPLE_CONCURRENCY}")
    print_simple(f"📁 保存路径: {SAVE_FILE}")
    print_simple(f"🔧 基础轮数：{BASE_MAX_ROUND//2} | 最大追加轮数：{MAX_APPEND_ROUND}")
    
    # 步骤1：初始化JSON文件（写入数组开头）
    try:
        with WRITE_LOCK, open(SAVE_FILE, "w", encoding="utf-8") as f:
            f.write("[\n")  # JSON数组开头
        print_simple(f"✅ 初始化输出文件成功：{SAVE_FILE}")
    except Exception as e:
        print_simple(f"❌ 初始化文件失败：{str(e)}")
        sys.exit(1)
    
    # 步骤2：并发生成样本，生成一个写入一个
    with ThreadPoolExecutor(max_workers=MAX_SAMPLE_CONCURRENCY, thread_name_prefix="SampleGen") as executor:
        # 提交所有样本生成任务
        future_to_id = {executor.submit(build_sample, i): i for i in range(n)}
        
        # 处理完成的任务，逐个写入文件
        for future in as_completed(future_to_id):
            sample_id = future_to_id[future]
            try:
                sample = future.result()
                if sample is not None:
                    # 线程安全写入当前样本
                    with WRITE_LOCK:
                        with open(SAVE_FILE, "a", encoding="utf-8") as f:
                            sample_json = json.dumps(sample, ensure_ascii=False, indent=2)
                            if IS_FIRST_SAMPLE:
                                # 第一个样本直接写入
                                f.write(sample_json)
                                IS_FIRST_SAMPLE = False
                            else:
                                # 后续样本先写逗号再写内容（保证JSON格式合法）
                                f.write(",\n")
                                f.write(sample_json)
                        print_simple(f"📝 样本{sample_id+1}已成功写入文件")
            except Exception as e:
                error_msg = f"样本{sample_id+1}任务异常：{str(e)[:50]}"
                print_simple(f"❌ {error_msg}")
                with WRITE_LOCK:
                    ERROR_SAMPLES.append(sample_id)
    
    # 步骤3：闭合JSON数组（写入结尾符）
    try:
        with WRITE_LOCK, open(SAVE_FILE, "a", encoding="utf-8") as f:
            f.write("\n]")  # JSON数组结尾
        print_simple(f"✅ 闭合JSON数组完成，文件写入结束")
    except Exception as e:
        print_simple(f"❌ 闭合JSON数组失败：{str(e)}")
        sys.exit(1)
    
    # 打印最终统计
    print_simple("\n" + "="*80)
    print_simple(f"✅ 并发生成完成！")
    print_simple(f"📊 总样本数：{n} | 成功：{n - len(ERROR_SAMPLES)} | 失败：{len(ERROR_SAMPLES)}")
    if ERROR_SAMPLES:
        print_simple(f"❌ 失败样本ID：{[i+1 for i in ERROR_SAMPLES]}")
    print_simple(f"📁 结果文件：{SAVE_FILE}")

if __name__ == "__main__":
    # 生成1690条样本（并发数10）
    main(1690)