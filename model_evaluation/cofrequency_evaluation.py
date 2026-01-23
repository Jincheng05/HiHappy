import json
import asyncio
import re
import time
import os
from datetime import datetime
from collections import defaultdict
from openai import AsyncOpenAI
from typing import List, Dict


# Configuration
VLLM_BASE_URL = "http://localhost:6008/v1"
VLLM_MODEL_NAME = "/mnt/nvme1n1/wjc/Model/Qwen2.5-7B-Instruct"
DATASET_PATH = "/mnt/nvme1n1/wjc/evaluation_res/MY/eval_Emo+four_data_on_test.json"
OUTPUT_PATH = "/mnt/nvme1n1/wjc/evaluation_res/emo-score/frequency_sync.json"
MAX_CONCURRENCY = 10
PREVIEW_LENGTH = 200
TOTAL_TASKS = 0
COMPLETED_TASKS = 0
COMPLETED_LOCK = asyncio.Lock()
WRITE_LOCK = asyncio.Lock()


# Helper functions
def print_with_timestamp(msg: str):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

async def update_progress():
    """Update progress asynchronously"""
    global COMPLETED_TASKS
    async with COMPLETED_LOCK:
        COMPLETED_TASKS += 1
    progress = (COMPLETED_TASKS / TOTAL_TASKS) * 100
    if COMPLETED_TASKS % 10 == 0 or COMPLETED_TASKS == TOTAL_TASKS:
        print_with_timestamp(f"📌 Progress: {COMPLETED_TASKS}/{TOTAL_TASKS} tasks completed ({progress:.1f}%)")

async def write_result_to_file(result: Dict):
    """Write evaluation result to file asynchronously"""
    async with WRITE_LOCK:
        try:
            if not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0:
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump([result], f, ensure_ascii=False, indent=2)
                print_with_timestamp(f"📝 First write | Sample{result['Sample_ID']}-Dialog{result['Dialog_ID']}-Turn{result['Turn_Number']}")
            else:
                with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                existing_data.append(result)
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                if COMPLETED_TASKS % 20 == 0:
                    print_with_timestamp(f"📝 Appended | {COMPLETED_TASKS} tasks completed")
        except Exception as e:
            print_with_timestamp(f"❌ Write failed | {str(e)[:100]}")


# Evaluation prompt template
PROMPT_TEMPLATE = """
### Core Instructions
1. Strictly follow the 0-3 point evaluation criteria for co-frequency ability;
2. Analysis must cover four dimensions: emotion capture, communication rhythm, boundary respect, and implicit needs.

### Evaluation Role
You are a professional evaluator with deep knowledge in psychology and linguistics, focusing on evaluating counselors' co-frequency ability in psychological counseling scenarios.

### Evaluation Goal
Evaluate the counselor's co-frequency ability - whether they can accurately capture the client's emotional state, match communication rhythm, respect boundaries, and understand implicit needs, achieving "emotional synchronization, rhythm alignment, and need awareness".

### Evaluation Dimensions and 0-3 Point Scoring Criteria
Co-frequency ability includes 4 core dimensions (emotion capture 40%, communication rhythm 30%, boundary respect 20%, implicit needs 10%):

0 points (Unqualified): Serious reverse behaviors exist, co-frequency ability completely absent. Denying/downplaying client emotions, oppressive interaction, breaking comfort zone, completely unable to identify implicit needs.

1 point (Qualified): No serious reverse behaviors, but obvious deficiencies in multiple dimensions. Can identify explicit emotions but frequently misaligned, occasional rhythm disconnection, basic boundary respect but occasional coercive language, almost unable to identify implicit needs.

2 points (Good): No reverse behaviors in core dimensions, minor deficiencies in few dimensions. Accurately identify explicit emotions, communication rhythm basically matched, fully respect boundaries, only occasional imprecision in implicit needs capture.

3 points (Excellent): All 4 dimensions without reverse behaviors, outstanding positive performance. Accurately identify explicit/implicit emotions, perfectly match communication rhythm, ultimate boundary respect, accurately capture implicit needs.

### Reference Materials
《Conversation History》:
{conversation_history}

《Model Generated Response》:
{generated_response}

### Output Format (strictly follow)
Co-frequency Score: 0/1/2/3
Analysis: [Detailed explanation of the reasoning process for the score, covering all 4 dimensions]
"""


# Data processing and evaluation logic
def load_and_group_dialogs(dataset_path: str) -> Dict[tuple, List[Dict]]:
    """Load dataset and group by Sample_ID + Dialog_ID"""
    print_with_timestamp(f"Loading dataset: {dataset_path}")
    start_time = time.time()
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print_with_timestamp(f"Dataset loaded | {len(dataset)} turn records | Time: {time.time()-start_time:.2f}s")
    
    grouped_dialogs = defaultdict(list)
    for turn in dataset:
        key = (turn["样本ID"], turn["对话ID"])
        grouped_dialogs[key].append(turn)
    
    for key in grouped_dialogs:
        grouped_dialogs[key].sort(key=lambda x: x["轮次号"])
    
    print_with_timestamp(f"Dataset grouped | {len(grouped_dialogs)} multi-turn dialogs")
    return grouped_dialogs


def prepare_evaluation_tasks(grouped_data: Dict[tuple, List[Dict]]) -> List[Dict]:
    """Prepare single-turn evaluation tasks"""
    print_with_timestamp("Building evaluation tasks...")
    tasks = []
    for (sample_id, dialog_id), turns in grouped_data.items():
        for idx, current_turn in enumerate(turns):
            history_turns = turns[:idx]
            conversation_history = []
            for h_turn in history_turns:
                conversation_history.append(f"User: {h_turn['当前用户问题']}")
                conversation_history.append(f"Assistant: {h_turn['完整参考回答']}")
            history_str = "\n".join(conversation_history) if history_turns else "No history"

            tasks.append({
                "Sample_ID": sample_id,
                "Dialog_ID": dialog_id,
                "Turn_Number": current_turn["轮次号"],
                "conversation_history": history_str,
                "reference_answer": current_turn["完整参考回答"],
                "generated_response": current_turn["完整生成回答"]
            })
    
    global TOTAL_TASKS
    TOTAL_TASKS = len(tasks)
    print_with_timestamp(f"Tasks built | {TOTAL_TASKS} evaluation tasks")
    return tasks


async def evaluate_single_turn(
    task: Dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore
) -> Dict:
    """Evaluate single turn asynchronously"""
    async with semaphore:
        sample_id = task["Sample_ID"]
        dialog_id = task["Dialog_ID"]
        turn_id = task["Turn_Number"]
        
        prompt = PROMPT_TEMPLATE.format(
            conversation_history=task["conversation_history"],
            generated_response=task["generated_response"]
        )

        response = await client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=2.0,
            max_tokens=600,
            stop=["🧠"]
        )
        eval_content = response.choices[0].message.content.strip()

        score_match = re.search(r"Co-frequency Score：(\d)", eval_content)
        analysis_match = re.search(r"Analysis：(.*)", eval_content, re.DOTALL)
        
        if not score_match or not analysis_match:
            raise ValueError(f"Output format error: {eval_content[:PREVIEW_LENGTH]}...")
        
        score = int(score_match.group(1))
        analysis = analysis_match.group(1).strip()
        
        if COMPLETED_TASKS % 20 == 0:
            print_with_timestamp(f"✅ Success | Sample{sample_id}-Dialog{dialog_id}-Turn{turn_id} | Score: {score}")
        
        result = {
            "Sample_ID": sample_id,
            "Dialog_ID": dialog_id,
            "Turn_Number": turn_id,
            "Cofrequency_Score": score,
            "Analysis": analysis
        }
        await write_result_to_file(result)
        await update_progress()
        
        return result


async def main():
    """Main process: load data → build tasks → async evaluation → statistics"""
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        print_with_timestamp(f"🗑️ Cleared output file: {OUTPUT_PATH}")
    
    total_start_time = time.time()
    
    grouped_dialogs = load_and_group_dialogs(DATASET_PATH)
    eval_tasks = prepare_evaluation_tasks(grouped_dialogs)
    
    if not eval_tasks:
        print_with_timestamp("⚠️ No valid tasks, exiting")
        return

    print_with_timestamp(f"Initializing VLLM client | URL: {VLLM_BASE_URL} | Model: {VLLM_MODEL_NAME}")
    client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="dummy-key"
    )

    print_with_timestamp(f"Starting evaluation | Max concurrency: {MAX_CONCURRENCY} | Total tasks: {TOTAL_TASKS}")
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [evaluate_single_turn(t, client, semaphore) for t in eval_tasks]
    await asyncio.gather(*tasks)

    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        success_count = sum(1 for r in all_results if r["Cofrequency_Score"] != -1)
        fail_count = len(all_results) - success_count
        avg_score = sum(r["Cofrequency_Score"] for r in all_results if r["Cofrequency_Score"] != -1) / success_count if success_count > 0 else 0
    except:
        success_count = 0
        fail_count = 0
        avg_score = 0
    
    total_cost = time.time() - total_start_time
    print_with_timestamp(f"\n==================== Evaluation Complete ====================")
    print_with_timestamp(f"📊 Statistics:")
    print_with_timestamp(f"   Total tasks: {TOTAL_TASKS}")
    print_with_timestamp(f"   Success: {success_count} ({success_count/TOTAL_TASKS*100:.1f}%)")
    print_with_timestamp(f"   Failed: {fail_count} ({fail_count/TOTAL_TASKS*100:.1f}%)")
    print_with_timestamp(f"   Average co-frequency score: {avg_score:.2f} (out of 3)")
    print_with_timestamp(f"⏱️  Time: {total_cost:.2f}s | Avg per task: {total_cost/TOTAL_TASKS:.2f}s")
    print_with_timestamp(f"📁 Results: {OUTPUT_PATH}")


if __name__ == "__main__":
    # Start vllm service: python -m vllm.serve.openai --model /path/to/model --port 6008
    asyncio.run(main())
