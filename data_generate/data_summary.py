#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
import time
import requests
import threading
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = "/mnt/nvme1n1/wjc/dataset_no_pre/dataset"
OUTPUT_PATH = "/mnt/nvme1n1/wjc/dataset_no_pre/perprepared/train_concurrency_en_cot.jsonl"

VLLM_URL = "http://localhost:6006/v1/chat/completions"
VLLM_MODEL = "/mnt/nvme1n1/wjc/Model/Qwen2.5-7B-Instruct"

MAX_NEW_TOKENS = 4048
TEMPERATURE = 0.3
TOP_P = 1.0
RETRY_TIMES = 3
TIMEOUT = 30
PREVIEW_LENGTH = 3000

MAX_CONCURRENCY = 15
WRITE_LOCK = threading.Lock()
def print_simple(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def build_dialog_text(messages):
    parts = []
    noise_keywords = ["和你聊天", "吃饭", "在吗", "测试", "无效内容", "占位符"]
    for m in messages:
        role = m.get("role", "未知角色")
        content = (m.get("content") or "").strip()
        
        if (not content 
            or len(content) < 3 
            or role == "system" 
            or any(keyword in content for keyword in noise_keywords)):
            continue
        
        parts.append(f"【原始角色：{role}】{content}")
    
    return "\n".join(parts)

def clean_extraction_result(extraction: str) -> Optional[str]:
    if not extraction:
        return None
    
    json_start = extraction.find("{")
    json_end = extraction.rfind("}")
    if json_start != -1:
        if json_end == -1:
            extraction = extraction[json_start:] + "}"
        else:
            extraction = extraction[json_start:json_end+1].strip()
    
    return extraction

def call_vllm(url: str, model: str, messages: List[Dict], max_tokens: int = 1024) -> Optional[str]:
    stop_words = ["}"]
    
    body = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
        "stream": False,
        "presence_penalty": 1.0,
        "frequency_penalty": 1.0,
    }

    for retry in range(RETRY_TIMES):
        try:
            resp = requests.post(url, json=body, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            else:
                print_simple(f"⚠️ VLLM request failed: status {resp.status_code}, retry {retry+1}")
        except Exception as e:
            print_simple(f"⚠️ VLLM call error: {str(e)[:50]}, retry {retry+1}")
            time.sleep(2 * (retry + 1))
    
    return None

def call_chatmodeldp(dialog_text: str) -> Optional[str]:
    system_prompt = (
        "你是专业的心理咨询对话信息抽取专家，需严格遵守以下指令完成信息提取，任何违规输出均视为无效：\n"
        "【格式铁律（必须100%遵守）】\n"
        "1. 输出内容仅包含完整的JSON字符串，无任何前置、后置文字（如“好的”“以下是结果”等）；\n"
        "2. JSON必须以`{`开头、`}`结尾，结构完整闭合，缺失结尾`}`直接判定为输出错误；\n"
        "3. JSON内每个字段的值均为中文文本，无语法错误、无省略号，标点使用规范。\n"
        "【内容创作授权（允许润色+添加相关内容）】\n"
        "1. 核心原则：基于对话中已有的真实信息，可充分润色语言（使表述更流畅,但是需要口语化），并合理添加**符合逻辑的相关内容**（禁止编造与对话无关的核心事实，如未提及的成绩、事件、人物等）；\n"
        "2. 可润色/添加的内容类型：\n"
        "   - 心理层面：补充孩子行为背后的潜在主观感受（如“叛逆行为背后实则是渴望被家长理解，而非单纯对抗”）；\n"
        "   - 场景细节：补充同类问题的典型表现（如“初一阶段孩子正处于青春期早期，叛逆行为常体现为拒绝沟通、刻意疏远家长”）；\n"
        "   - 影响延伸：补充问题可能带来的潜在影响（如“长期消极厌学若未干预，可能导致后续学习动力持续下降”）；\n"
        "   - 语言优化：将口语化表述转为专业、流畅的书面语，补充连接词/逻辑词使内容更连贯；\n"
        "3. 「主要困扰」：字数不少于120字，必须包含：问题核心+持续时长（精确到天/周/月）+ 不同场景下的具体表现 + 孩子的主观感受 + 实际影响；可润色语言并添加行为背后的心理动机、同类问题的典型特征；\n"
        "4. 「诱因或重要事件」：字数不少于100字，必须包含：事件时间+具体场景+完整过程+即时反应+关联逻辑；可润色事件描述的连贯性，并添加事件对孩子心理状态的潜在影响；\n"
        "【角色与提取范围】\n"
        "对话标注【原始角色：xxx】（标识可能错误），先判断真实角色：\n"
        "- 来访者/家属：主动描述孩子问题、自身困扰的参与方；\n"
        "- 咨询师：回应、询问情况的参与方。\n"
        "仅提取【被咨询的孩子（核心来访者）】的关键信息，按以下维度总结（未提及则填“未提及”）：\n"
        "1. 基本情况：年龄（精确到岁）、性别、年级/身份、日常作息/兴趣等基础信息；\n"
        "2. 主要困扰：按上述润色/添加要求，完整描述孩子核心问题及背景；\n"
        "3. 症状表现：情绪（如焦虑的具体触发点/频率）、躯体不适（具体部位/发作时间）、睡眠（入睡时长/夜醒次数）、学习/社交（具体影响程度）；\n"
        "4. 诱因或重要事件：按上述润色/添加要求，完整还原引发问题的关键事件；\n"
        "5. 其他信息：家庭关系（如和父母的沟通频率）、学校表现（如最近一次考试排名）、过往类似情况等细节。\n"
        "【标准示例（必须严格遵守格式+润色逻辑）】\n"
        "{\n"
        "  \"基本情况\": \"17岁，女，高三文科班学生，日常每天学习约10小时，无明显兴趣爱好，周末基本在家复习；高三阶段学业压力陡增，是该年龄段学生焦虑情绪高发的典型时期\",\n"
        "  \"主要困扰\": \"长期存在考试焦虑问题，该情况从高二下学期（约8个月前）开始出现，且近1个月愈发严重；在校期间只要临近模考就会坐立不安，无法集中精力刷题，甚至会躲在卫生间哭泣，在家复习时看到试卷就会手抖、心跳加速，主观上觉得“如果考不好就对不起父母”——这种负罪感实则是高三学生面对升学压力时的典型心理反应；该状态导致最近三次模考数学成绩从120分左右下降到85分，且不愿和同学讨论考试相关话题，社交活动几乎完全停止，进一步加剧了其孤独感和焦虑感\",\n"
        "  \"症状表现\": \"每周至少3次出现莫名的心慌，傍晚时段尤为明显（该时段是学生复盘当日学习效果、压力感最强的阶段）；入睡需要1.5小时以上，每晚至少夜醒2次，晨起有头晕、乏力的躯体不适；上课无法专注听讲，笔记记录混乱，作业完成效率较之前下降60%，形成“成绩下滑-焦虑加重”的恶性循环\",\n"
        "  \"诱因或重要事件\": \"本次焦虑加重的直接诱因是2个月前的全市统考，孩子原本目标是年级前50名，但最终只考了年级120名；考试当天孩子因紧张漏做了两道大题，出考场后被母亲指责“不够努力”，回家后和母亲发生激烈争吵，当晚失眠至凌晨3点；该事件不仅让孩子产生了强烈的自我否定，认为“自己永远达不到父母的要求”，更让其将考试失败与“自身价值”绑定，进而导致每次看到试卷就触发焦虑情绪，形成恶性循环\",\n"
        "【最终提醒】JSON必须完整闭合（含最后的引号`\"`和结尾符`}`），内容可充分润色并添加逻辑相关细节，但所有补充内容需贴合对话已有信息，不得编造核心事实。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": dialog_text},
    ]

    extraction = call_vllm(
        url=VLLM_URL,
        model=VLLM_MODEL,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS
    )
    if extraction is not None:
        extraction = clean_extraction_result(extraction)
    
    return extraction
def process_single_sample(sample: Dict, source_file: str, out_f_path: str) -> Dict:
    item_id = sample["id"]
    messages = sample["messages"]
    result_obj = {
        "source_file": source_file,
        "id": item_id,
        "extraction": None,
        "is_valid_json": False,
        "error": None,
        "status": "failed"
    }

    try:
        dialog_text = build_dialog_text(messages)
        if not dialog_text.strip():
            result_obj["error"] = "Empty dialog, skipped"
            print_simple(f"📌 Sample ID: {item_id} | {result_obj['error']}")
            return result_obj

        extraction = call_chatmodeldp(dialog_text)
        result_obj["extraction"] = extraction

        print_simple("-" * 80)
        print_simple(f"📌 Processing sample | Source: {source_file} | ID: {item_id}")
        
        if extraction is None:
            result_obj["error"] = "VLLM call failed after retries"
            print_simple(f"❌ Extraction failed | Reason: {result_obj['error']}")
        else:
            try:
                json.loads(extraction)
                result_obj["is_valid_json"] = True
                result_obj["status"] = "success"
                is_valid = "✅ Valid JSON"
            except json.JSONDecodeError:
                result_obj["is_valid_json"] = False
                result_obj["error"] = "Invalid JSON format"
                is_valid = "❌ Invalid JSON"
            
            preview_content = extraction[:PREVIEW_LENGTH] + "..." if len(extraction) > PREVIEW_LENGTH else extraction
            print_simple(f"📝 Result ({is_valid}) | Preview:\n{preview_content}")
        
        print_simple("-" * 80 + "\n")

        with WRITE_LOCK:
            with open(out_f_path, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                out_f.flush()

    except Exception as e:
        error_msg = f"Sample processing error: {str(e)[:100]}"
        result_obj["error"] = error_msg
        result_obj["status"] = "error"
        result_obj["is_valid_json"] = False
        print_simple(f"❌ Sample ID: {item_id} | {error_msg}")

    return result_obj

def process_single_file_concurrent(path: str, out_f_path: str) -> Dict:
    print_simple(f"[info] Processing file: {path}")
    file_stats = {
        "file": path,
        "total_samples": 0,
        "valid_samples": 0,
        "success": 0,
        "failed": 0,
        "error": 0,
        "invalid_json_count": 0,
        "errors": []
    }

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"JSON parse failed: pos {e.pos} | {e.msg}"
        print_simple(f"❌ {error_msg}")
        file_stats["error"] += 1
        file_stats["errors"].append(error_msg)
        return file_stats
    except Exception as e:
        error_msg = f"File read failed: {str(e)[:50]}"
        print_simple(f"❌ {error_msg}")
        file_stats["error"] += 1
        file_stats["errors"].append(error_msg)
        return file_stats

    records = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "id" in data and "messages" in data:
        records = [data]
    elif isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list):
                records.extend(val)
            elif isinstance(val, dict) and "id" in val and "messages" in val:
                records.append(val)

    valid_records = []
    for idx, item in enumerate(records):
        if (isinstance(item, dict) 
            and "id" in item 
            and "messages" in item 
            and isinstance(item["messages"], list)):
            valid_records.append(item)
        else:
            error_msg = f"Sample {idx} invalid (missing id/messages)"
            print_simple(f"⚠️ {error_msg}")
            file_stats["errors"].append(error_msg)

    file_stats["total_samples"] = len(records)
    file_stats["valid_samples"] = len(valid_records)

    if not valid_records:
        print_simple(f"[warn] No valid samples in file, skipped: {path}")
        return file_stats

    source_file_name = os.path.basename(path)
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        for sample in valid_records:
            future = executor.submit(
                process_single_sample,
                sample=sample,
                source_file=source_file_name,
                out_f_path=out_f_path
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result()
                if result["status"] == "success":
                    file_stats["success"] += 1
                elif result["status"] == "failed":
                    file_stats["failed"] += 1
                    if not result["is_valid_json"] and result["extraction"] is not None:
                        file_stats["invalid_json_count"] += 1
                elif result["status"] == "error":
                    file_stats["error"] += 1
                
                if result["error"]:
                    file_stats["errors"].append(f"Sample {result['id']}: {result['error']}")
            except Exception as e:
                error_msg = f"Task execution error: {str(e)[:50]}"
                file_stats["error"] += 1
                file_stats["errors"].append(error_msg)

    print_simple(f"[info] File completed: {path} | Valid: {len(valid_records)} | Success: {file_stats['success']} | Failed: {file_stats['failed']} | Error: {file_stats['error']} | Invalid JSON: {file_stats['invalid_json_count']}")
    return file_stats
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("")

    json_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if not json_files:
        print_simple(f"[error] No JSON files found in {DATA_DIR}")
        return
    
    global_stats = {
        "total_files": len(json_files),
        "processed_files": 0,
        "total_samples": 0,
        "valid_samples": 0,
        "success": 0,
        "failed": 0,
        "error": 0,
        "invalid_json_count": 0
    }

    print_simple(f"[info] Found {len(json_files)} JSON files to process")
    print_simple(f"[info] Max concurrency: {MAX_CONCURRENCY}")
    print_simple(f"[info] Preview length: {PREVIEW_LENGTH} chars")
    print_simple(f"[info] Output: {OUTPUT_PATH}\n")

    for file_idx, path in enumerate(json_files, 1):
        print_simple(f"\n" + "="*100)
        print_simple(f"[info] Processing file {file_idx}/{len(json_files)}: {path}")
        file_stats = process_single_file_concurrent(path, OUTPUT_PATH)
        
        global_stats["processed_files"] += 1
        global_stats["total_samples"] += file_stats["total_samples"]
        global_stats["valid_samples"] += file_stats["valid_samples"]
        global_stats["success"] += file_stats["success"]
        global_stats["failed"] += file_stats["failed"]
        global_stats["error"] += file_stats["error"]
        global_stats["invalid_json_count"] += file_stats["invalid_json_count"]

    print_simple("\n" + "="*100)
    print_simple("✅ All files processed! Final statistics:")
    print_simple(f"📊 Total files: {global_stats['total_files']} | Processed: {global_stats['processed_files']}")
    print_simple(f"📊 Total samples: {global_stats['total_samples']} | Valid: {global_stats['valid_samples']}")
    print_simple(f"📊 Success: {global_stats['success']} | Failed: {global_stats['failed']} | Error: {global_stats['error']}")
    print_simple(f"📊 Invalid JSON: {global_stats['invalid_json_count']}")
    print_simple(f"📊 Output file: {OUTPUT_PATH}")
    total_processed = global_stats["success"] + global_stats["failed"] + global_stats["error"]
    print_simple(f"📊 Total processed: {total_processed} (should equal valid: {global_stats['valid_samples']})")

if __name__ == "__main__":
    main()