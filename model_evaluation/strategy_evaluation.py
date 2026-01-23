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
VLLM_BASE_URL = "http://localhost:6007/v1"
VLLM_MODEL_NAME = "/mnt/nvme1n1/wjc/Model/Qwen2.5-7B-Instruct"
DATASET_PATH = "/mnt/nvme1n1/wjc/evaluation_res/MY/eval_Emo+four_data_on_test2.json"
OUTPUT_PATH = "/mnt/nvme1n1/wjc/evaluation_res/emo-score/dialog_strategy1.json"
MAX_CONCURRENCY = 50
TOTAL_TASKS = 0
COMPLETED_TASKS = 0
COMPLETED_LOCK = asyncio.Lock()
WRITE_LOCK = asyncio.Lock()


def print_with_timestamp(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

async def update_progress():
    global COMPLETED_TASKS
    async with COMPLETED_LOCK:
        COMPLETED_TASKS += 1

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
            print_with_timestamp(f"❌ Write error: {str(e)[:100]}")


# Dialogue strategy evaluation prompt
PROMPT_TEMPLATE = """
### Evaluation Role
Professional evaluator focusing on dialogue strategy effectiveness in psychological counseling.

### Evaluation Goal
Evaluate the dialogue strategy dimension of counseling responses based on conversation history.

### Evaluation Dimension: Dialogue Strategy
Definition: Core techniques and methods used in counseling communication, including questioning & exploration, feedback & summary, and problem-solving & guidance.

### Scoring Criteria (0-3 points)
0 points: No effective strategy. Inappropriate strategies (preaching, denial) or no strategy when needed.

1 point: Very limited strategy. Only partial appropriate strategies (simple questions), but ineffective in guiding self-reflection.

2 points: Moderate strategy. Appropriate strategies used (questioning/feedback/guidance), effective in guiding thinking, but may be too direct or slightly directive.

3 points: Highly effective strategy. Strategies precisely match the scenario, effectively guide self-reflection, maintain dialogue coherence, fully support autonomous decision-making.

### Reference Materials
《Conversation History》:
{conversation_history}

《Model Generated Response》:
{generated_response}

### Output Format
Dialogue Strategy Score: 0/1/2/3
Analysis: [Explain strategy appropriateness and guidance effectiveness]
"""


def load_and_group_dialogs(dataset_path: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    grouped = defaultdict(list)
    for turn in dataset:
        grouped[(turn["样本ID"], turn["对话ID"])].append(turn)
    for key in grouped:
        grouped[key].sort(key=lambda x: x["轮次号"])
    return grouped


def prepare_evaluation_tasks(grouped_data):
    tasks = []
    for (sid, did), turns in grouped_data.items():
        for idx, turn in enumerate(turns):
            history = []
            for h in turns[:idx]:
                history.append(f"User: {h['当前用户问题']}")
                history.append(f"Assistant: {h['完整参考回答']}")
            tasks.append({
                "Sample_ID": sid,
                "Dialog_ID": did,
                "Turn_Number": turn["轮次号"],
                "conversation_history": "\n".join(history) if history else "No history",
                "generated_response": turn["完整生成回答"]
            })
    global TOTAL_TASKS
    TOTAL_TASKS = len(tasks)
    return tasks


async def evaluate_single_turn(task, client, semaphore):
    async with semaphore:
        prompt = PROMPT_TEMPLATE.format(
            conversation_history=task["conversation_history"],
            generated_response=task["generated_response"]
        )
        try:
            response = await client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.9,
                max_tokens=600,
                stop=["🧠"]
            )
            content = response.choices[0].message.content.strip()
            score = int(re.search(r"Dialogue Strategy Score：(\d)", content).group(1))
            analysis = re.search(r"Analysis：(.*)", content, re.DOTALL).group(1).strip()
            
            result = {
                "Sample_ID": task["Sample_ID"],
                "Dialog_ID": task["Dialog_ID"],
                "Turn_Number": task["Turn_Number"],
                "Strategy_Score": score,
                "Analysis": analysis
            }
        except Exception as e:
            result = {
                "Sample_ID": task["Sample_ID"],
                "Dialog_ID": task["Dialog_ID"],
                "Turn_Number": task["Turn_Number"],
                "Strategy_Score": -1,
                "Analysis": f"Failed: {str(e)}"
            }
        
        await write_result_to_file(result)
        await update_progress()
        return result


async def main():
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    
    grouped = load_and_group_dialogs(DATASET_PATH)
    tasks = prepare_evaluation_tasks(grouped)
    
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="dummy-key")
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    await asyncio.gather(*[evaluate_single_turn(t, client, semaphore) for t in tasks])
    
    print_with_timestamp("Evaluation complete")


if __name__ == "__main__":
    asyncio.run(main())
