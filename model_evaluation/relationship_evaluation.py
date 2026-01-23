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
VLLM_BASE_URL = "http://localhost:6006/v1"
VLLM_MODEL_NAME = "/mnt/nvme1n1/wjc/Model/Qwen2.5-7B-Instruct"
DATASET_PATH = "/mnt/nvme1n1/wjc/evaluation_res/MY/eval_Emo+four_data_on_test2.json"
OUTPUT_PATH = "/mnt/nvme1n1/wjc/evaluation_res/emo-score/relationship_building.json"
MAX_CONCURRENCY = 50
PREVIEW_LENGTH = 200
TOTAL_TASKS = 0
COMPLETED_TASKS = 0
COMPLETED_LOCK = asyncio.Lock()
WRITE_LOCK = asyncio.Lock()


def print_with_timestamp(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

async def update_progress():
    global COMPLETED_TASKS
    async with COMPLETED_LOCK:
        COMPLETED_TASKS += 1
    progress = (COMPLETED_TASKS / TOTAL_TASKS) * 100
    if COMPLETED_TASKS % 10 == 0 or COMPLETED_TASKS == TOTAL_TASKS:
        print_with_timestamp(f"📌 Progress: {COMPLETED_TASKS}/{TOTAL_TASKS} ({progress:.1f}%)")

async def write_result_to_file(result: Dict):
    async with WRITE_LOCK:
        try:
            if not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0:
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump([result], f, ensure_ascii=False, indent=2)
            else:
                with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                existing_data.append(result)
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print_with_timestamp(f"❌ Write failed: {str(e)[:100]}")


# Relationship building evaluation prompt
PROMPT_TEMPLATE = """
### Core Instructions
1. No thinking process or extra explanations, only output the specified format;
2. Strictly follow the relationship building evaluation criteria;
3. Avoid professional terminology, use plain language.

### Evaluation Role
Professional evaluator with deep psychology and linguistics knowledge, focusing on relationship building ability in psychological counseling.

### Evaluation Goal
Evaluate the model's ability to build safe, trusting, and collaborative counseling relationships through responses.

### Evaluation Dimension: Relationship Building
Definition: The core ability to build safe, trusting, and collaborative counseling relationships through responses, making clients feel accepted, valued, and safe.

### Scoring Criteria (0-3 points)
0 points: No relationship building. Response completely ignores relationship construction, mechanical answers only, no acceptance/value/safety expression.

1 point: Very limited relationship building. Only surface-level attempts (like "I understand you"), cannot create safety, client still has obvious defensive psychology.

2 points: Moderate relationship building. Can make client feel basic acceptance and value, create initial safety, establish shallow trust, client willing to express surface needs.

3 points: High relationship building. Precisely adapt to client state, create safety and trust through responses, make client feel fully accepted, willing to open core needs.

### Reference Materials
《Conversation History》:
{conversation_history}

《Model Generated Response》:
{generated_response}

### Output Format
Relationship Building Score: 0/1/2/3
Analysis: [Detailed explanation covering trust building, safety creation, and acceptance expression]
"""


def load_and_group_dialogs(dataset_path: str) -> Dict[tuple, List[Dict]]:
    print_with_timestamp(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    grouped_dialogs = defaultdict(list)
    for turn in dataset:
        key = (turn["样本ID"], turn["对话ID"])
        grouped_dialogs[key].append(turn)
    
    for key in grouped_dialogs:
        grouped_dialogs[key].sort(key=lambda x: x["轮次号"])
    
    return grouped_dialogs


def prepare_evaluation_tasks(grouped_data: Dict[tuple, List[Dict]]) -> List[Dict]:
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
                "generated_response": current_turn["完整生成回答"]
            })
    
    global TOTAL_TASKS
    TOTAL_TASKS = len(tasks)
    return tasks


async def evaluate_single_turn(task: Dict, client: AsyncOpenAI, semaphore: asyncio.Semaphore) -> Dict:
    async with semaphore:
        prompt = PROMPT_TEMPLATE.format(
            conversation_history=task["conversation_history"],
            generated_response=task["generated_response"]
        )

        try:
            response = await client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=600,
                stop=["🧠"]
            )
            eval_content = response.choices[0].message.content.strip()

            score_match = re.search(r"Relationship Building Score：(\d)", eval_content)
            analysis_match = re.search(r"Analysis：(.*)", eval_content, re.DOTALL)
            
            if not score_match or not analysis_match:
                raise ValueError(f"Format error: {eval_content[:PREVIEW_LENGTH]}")
            
            result = {
                "Sample_ID": task["Sample_ID"],
                "Dialog_ID": task["Dialog_ID"],
                "Turn_Number": task["Turn_Number"],
                "Relationship_Score": int(score_match.group(1)),
                "Analysis": analysis_match.group(1).strip()
            }
            await write_result_to_file(result)
            await update_progress()
            return result
        
        except Exception as e:
            fail_result = {
                "Sample_ID": task["Sample_ID"],
                "Dialog_ID": task["Dialog_ID"],
                "Turn_Number": task["Turn_Number"],
                "Relationship_Score": -1,
                "Analysis": f"Evaluation failed: {str(e)}"
            }
            await write_result_to_file(fail_result)
            await update_progress()
            return fail_result


async def main():
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    
    grouped_dialogs = load_and_group_dialogs(DATASET_PATH)
    eval_tasks = prepare_evaluation_tasks(grouped_dialogs)
    
    if not eval_tasks:
        return

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="dummy-key")
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [evaluate_single_turn(t, client, semaphore) for t in eval_tasks]
    await asyncio.gather(*tasks)

    print_with_timestamp("Evaluation complete")


if __name__ == "__main__":
    asyncio.run(main())
