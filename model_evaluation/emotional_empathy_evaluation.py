import json
import re
import time
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# Configuration
VLLM_URL = "http://localhost:6006/v1/chat/completions"
VLLM_MODEL_NAME = "/mnt/nvme1n1/wjc/Model/Qwen2.5-7B-Instruct"
DATASET_PATH = "/mnt/nvme1n1/wjc/evaluation_res/MY/eval_Emo+four_data_on_test.json"
OUTPUT_PATH = "/mnt/nvme1n1/wjc/evaluation_res/emo-score/emotion_empathy2.json"
MAX_NEW_TOKENS = 1600
TEMPERATURE = 0.2
TOP_P = 1.0
RETRY_TIMES = 3
TIMEOUT = 30
MAX_CONCURRENCY = 10
WRITE_LOCK = threading.Lock()
TOTAL_TASKS = 0
COMPLETED_TASKS = 0
COMPLETED_LOCK = threading.Lock()


def print_simple(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def update_completed_count():
    global COMPLETED_TASKS
    with COMPLETED_LOCK:
        COMPLETED_TASKS += 1
    progress = (COMPLETED_TASKS / TOTAL_TASKS) * 100
    print_simple(f"📌 Progress: {COMPLETED_TASKS}/{TOTAL_TASKS} ({progress:.1f}%)")

def load_and_group_dialogs(dataset_path: str):
    print_simple(f"📥 Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    grouped_dialogs = {}
    for turn in dataset:
        key = (turn.get("样本ID", "unknown"), turn.get("对话ID", "unknown"))
        if key not in grouped_dialogs:
            grouped_dialogs[key] = []
        grouped_dialogs[key].append(turn)
    
    for key in grouped_dialogs:
        grouped_dialogs[key].sort(key=lambda x: x.get("轮次号", 0))
    
    return grouped_dialogs

def prepare_evaluation_tasks(grouped_data):
    tasks = []
    for (sample_id, dialog_id), turns in grouped_data.items():
        for idx, current_turn in enumerate(turns):
            history = []
            for h_turn in turns[:idx]:
                history.append(f"User: {h_turn['当前用户问题']}")
                history.append(f"Assistant: {h_turn['完整参考回答']}")
            
            tasks.append({
                "Sample_ID": sample_id,
                "Dialog_ID": dialog_id,
                "Turn_Number": current_turn.get("轮次号", idx+1),
                "conversation_history": "\n".join(history) if history else "No history",
                "reference_answer": current_turn["完整参考回答"],
                "generated_response": current_turn["完整生成回答"]
            })
    
    global TOTAL_TASKS
    TOTAL_TASKS = len(tasks)
    return tasks

def call_vllm(messages: List[Dict]) -> Optional[str]:
    """Call VLLM API with retry"""
    body = {
        "model": VLLM_MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_NEW_TOKENS,
        "stream": False
    }

    for retry in range(RETRY_TIMES):
        try:
            resp = requests.post(VLLM_URL, json=body, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            time.sleep(2 * (retry + 1))
    return None

def evaluate_single_turn(task: Dict) -> Dict:
    """Evaluate single turn"""
    result_obj = {
        "Sample_ID": task["Sample_ID"],
        "Dialog_ID": task["Dialog_ID"],
        "Turn_Number": task["Turn_Number"],
        "Emotional_Empathy_Score": -1,
        "Analysis": "",
        "status": "failed"
    }

    try:
        PROMPT_TEMPLATE = """
### Evaluation Goal
Evaluate the emotional empathy of psychological counseling responses based on conversation history.

### Evaluation Dimension: Emotional Empathy
Definition: Ability to perceive and share others' emotions, including 4 core elements:
- Perception: Accurately identify client's emotional state
- Resonance: Generate similar emotional experience
- Understanding: Grasp reasons behind emotions
- Response: Give appropriate sympathetic/supportive feedback

### Scoring Criteria
0 points: No empathy. Cannot identify emotions, no resonance/understanding, cold response.
1 point: Limited empathy. Vague emotion perception, brief surface resonance, shallow understanding, passive response.
2 points: Moderate empathy. Accurate emotion perception, real but limited resonance, proactive caring response.
3 points: High empathy. Precise emotion capture, deep resonance, full understanding, consistently deep support.

### Reference Materials
《Conversation History》:
{conversation_history}

《Reference Answer》: {reference_answer}

《Model Generated Response》: {generated_response}

### Output Format
Emotional Empathy Score: 0/1/2/3
Analysis: [Detailed explanation of the score]
        """
        prompt = PROMPT_TEMPLATE.format(
            conversation_history=task["conversation_history"],
            reference_answer=task["reference_answer"],
            generated_response=task["generated_response"]
        )

        messages = [{"role": "user", "content": prompt}]
        eval_content = call_vllm(messages)
        
        if eval_content is None:
            result_obj["error"] = "VLLM call failed after retries"
            return result_obj

        score_match = re.search(r"Emotional Empathy Score：(\d)", eval_content)
        analysis_match = re.search(r"Analysis：(.*)", eval_content, re.DOTALL)
        
        if not score_match or not analysis_match:
            result_obj["error"] = "Cannot parse output format"
            return result_obj

        result_obj["Emotional_Empathy_Score"] = int(score_match.group(1))
        result_obj["Analysis"] = analysis_match.group(1).strip()
        result_obj["status"] = "success"

    except Exception as e:
        result_obj["error"] = f"Exception: {str(e)[:100]}"
        result_obj["status"] = "error"

    update_completed_count()
    return result_obj

def write_result_to_file(result: Dict, output_path: str):
    """Thread-safe write to file"""
    with WRITE_LOCK:
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([result], f, ensure_ascii=False, indent=2)
        else:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_data.append(result)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)

def process_evaluation_tasks_concurrent(eval_tasks: List[Dict], output_path: str):
    """Process evaluation tasks concurrently"""
    print_simple(f"🚀 Starting concurrent evaluation | Max concurrency: {MAX_CONCURRENCY}")
    
    if os.path.exists(output_path):
        os.remove(output_path)

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        for task in eval_tasks:
            future = executor.submit(evaluate_single_turn, task=task)
            futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result()
                write_result_to_file(result, output_path)
            except Exception as e:
                print_simple(f"❌ Task error: {str(e)[:50]}")

    print_simple(f"🏁 All tasks completed!")

def main():
    """Main process"""
    try:
        grouped_dialogs = load_and_group_dialogs(DATASET_PATH)
        eval_tasks = prepare_evaluation_tasks(grouped_dialogs)
        
        if not eval_tasks:
            print_simple("⚠️ No valid tasks")
            return
        
        process_evaluation_tasks_concurrent(eval_tasks, OUTPUT_PATH)
        print_simple(f"📁 Results saved to: {OUTPUT_PATH}")

    except Exception as e:
        print_simple(f"❌ Program error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
