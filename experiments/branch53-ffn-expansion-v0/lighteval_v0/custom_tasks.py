"""Frozen Lighteval 0.13.0 task definitions for the paired Helix court.

Prompt functions and metrics intentionally mirror the corresponding Lighteval
0.13.0 built-ins. Dataset revisions and gold-bearing evaluation splits are
explicit so upstream dataset movement cannot change the comparison.
"""

from __future__ import annotations

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def arc_prompt(line, task_name: str | None = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {choice}" for choice in line["choices"]["text"]],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )


def hellaswag_prompt(line, task_name: str | None = None):
    query = "The following are multiple choice questions (with answers) about common sense.\n\n"
    query += f"Question: {line['activity_label']}: {line['ctx_a']} {line['ctx_b'].capitalize()}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, line["endings"])])
    query += "Answer:"
    gold_index = int(line["label"]) if line["label"] != "" else -1
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + letter for letter in ascii_uppercase[: len(line["endings"])]],
        gold_index=gold_index,
        instruction="The following are multiple choice questions (with answers) about common sense.\n\n",
    )


def openbookqa_prompt(line, task_name: str | None = None):
    query = "The following are multiple choice questions (with answers) about common sense.\n"
    query += f"Question: {line['question_stem']}\n"
    query += "".join(
        [f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, line["choices"]["text"])]
    )
    query += "Answer: "
    return Doc(
        task_name=task_name,
        query=query,
        choices=list(ascii_uppercase[: len(line["choices"]["text"])]),
        gold_index=["A", "B", "C", "D", "E"].index(line["answerKey"].strip()),
        instruction="The following are multiple choice questions (with answers) about common sense.\n",
    )


def piqa_prompt(line, task_name: str | None = None):
    letters = list(ascii_uppercase)[:2]
    query = "The following are multiple choice questions (with answers) about common sense.\n"
    query += f"Question: {line['goal']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(letters, [line["sol1"], line["sol2"]])])
    query += "Answer: "
    is_few_shot = line.get("__few_shots", False)
    return Doc(
        task_name=task_name,
        query=query,
        choices=letters if not is_few_shot else [line["sol1"], line["sol2"]],
        gold_index=int(line["label"]),
        instruction="The following are multiple choice questions (with answers) about common sense.\n",
    )


def winogrande_prompt(line, task_name: str | None = None):
    query, end_of_target = line["sentence"].split("_")
    end_of_target = end_of_target.strip()
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"{line['option1']} {end_of_target}", f"{line['option2']} {end_of_target}"],
        gold_index=int(line["answer"]) - 1 if line["answer"] != "" else -1,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="helix_arc_easy",
        prompt_function=arc_prompt,
        hf_repo="allenai/ai2_arc",
        hf_revision="5a61ed188b8810abf32686fa83eb93787cbc5f25",
        hf_subset="ARC-Easy",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select="random_sampling_from_train",
        generation_size=1,
        metrics=[Metrics.loglikelihood_acc],
        stop_sequence=["\n"],
        version=0,
    ),
    LightevalTaskConfig(
        name="helix_hellaswag",
        prompt_function=hellaswag_prompt,
        hf_repo="Rowan/hellaswag",
        hf_revision="382ffb520660cc2ba98092cd8aa07cafc9c47f6d",
        hf_subset="default",
        hf_avail_splits=["train", "test", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=1,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        version=0,
    ),
    LightevalTaskConfig(
        name="helix_openbookqa",
        prompt_function=openbookqa_prompt,
        hf_repo="allenai/openbookqa",
        hf_revision="ff32d06f3b17a43c2042765b39a22ba55a0d41f9",
        hf_subset="main",
        hf_avail_splits=["train", "test", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=1,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        version=0,
    ),
    LightevalTaskConfig(
        name="helix_piqa",
        prompt_function=piqa_prompt,
        hf_repo="ybisk/piqa",
        hf_revision="142c51238b3ca2bc61e9a075913871b8b600e8e1",
        hf_subset="plain_text",
        hf_avail_splits=["train", "test", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=1,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
        version=0,
    ),
    LightevalTaskConfig(
        name="helix_winogrande",
        prompt_function=winogrande_prompt,
        hf_repo="allenai/winogrande",
        hf_revision="546d5aa0e42fc0fa9ad2c13260322ec969e083ee",
        hf_subset="winogrande_xl",
        hf_avail_splits=["train", "test", "validation"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        few_shots_select="random_sampling",
        generation_size=-1,
        metrics=[Metrics.loglikelihood_acc],
        stop_sequence=["\n"],
        version=0,
    ),
]
