# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally create just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""
import numpy as np
from aenum import extend_enum

from lighteval.metrics import Metrics
from lighteval.metrics.metrics import SampleLevelMetric
from lighteval.metrics.utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES, bbh


# EVAL WITH NO SUBSET ##
# This is how you create a simple tasks (like hellaswag) which has one single subset
# attached to it, and one evaluation possible.
# TODO: table and drcd
task = LightevalTaskConfig(
    name="tc-eval-v2:penguin_table",
    prompt_function="tceval_bbh_penguins_in_a_table",  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["community", "tc-eval-v2"],
    hf_repo="MediaTek-Research/TCEval-v2",
    hf_subset="penguin_table",
    hf_avail_splits=["test", "dev"],
    evaluation_splits=["test"],
    generation_size=20,
    metric=["exact_match","quasi_exact_match","prefix_exact_match","prefix_quasi_exact_match","perfect_exact_match"],
    stop_sequence=["</s>", "Q:", "\n\n"],
)

def tceval_bbh_penguins_in_a_table(line, task_name: str = None):
    instruction = "回答有關企鵝及其屬性的表格的問題。\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:5]]
    return bbh(line, instruction, choices, task_name)


# EVALS WITH SUBSET
# This is how you create a subset task (like MMLU), which has several subset
# each being its own evaluation task.

# fmt: off
SAMPLE_SUBSETS = ['tmmluplus-accounting', 'tmmluplus-administrative_law', 'tmmluplus-advance_chemistry', 'tmmluplus-agriculture', 'tmmluplus-anti_money_laundering', 'tmmluplus-auditing', 'tmmluplus-basic_medical_science', 'tmmluplus-business_management', 'tmmluplus-chinese_language_and_literature', 'tmmluplus-clinical_psychology', 'tmmluplus-computer_science', 'tmmluplus-culinary_skills', 'tmmluplus-dentistry', 'tmmluplus-economics', 'tmmluplus-education', 'tmmluplus-education_(profession_level)', 'tmmluplus-educational_psychology', 'tmmluplus-engineering_math', 'tmmluplus-finance_banking', 'tmmluplus-financial_analysis', 'tmmluplus-fire_science', 'tmmluplus-general_principles_of_law', 'tmmluplus-geography_of_taiwan', 'tmmluplus-human_behavior', 'tmmluplus-insurance_studies', 'tmmluplus-introduction_to_law', 'tmmluplus-jce_humanities', 'tmmluplus-junior_chemistry', 'tmmluplus-junior_chinese_exam', 'tmmluplus-junior_math_exam', 'tmmluplus-junior_science_exam', 'tmmluplus-junior_social_studies', 'tmmluplus-logic_reasoning', 'tmmluplus-macroeconomics', 'tmmluplus-management_accounting', 'tmmluplus-marketing_management', 'tmmluplus-mechanical', 'tmmluplus-music', 'tmmluplus-national_protection', 'tmmluplus-nautical_science', 'tmmluplus-occupational_therapy_for_psychological_disorders', 'tmmluplus-official_document_management', 'tmmluplus-optometry', 'tmmluplus-organic_chemistry', 'tmmluplus-pharmacology', 'tmmluplus-pharmacy', 'tmmluplus-physical_education', 'tmmluplus-physics', 'tmmluplus-politic_science', 'tmmluplus-real_estate', 'tmmluplus-secondary_physics', 'tmmluplus-statistics_and_machine_learning', 'tmmluplus-taiwanese_hokkien', 'tmmluplus-taxation', 'tmmluplus-technical', 'tmmluplus-three_principles_of_people', 'tmmluplus-trade', 'tmmluplus-traditional_chinese_medicine_clinical_medicine', 'tmmluplus-trust_practice', 'tmmluplus-ttqav2', 'tmmluplus-tve_chinese_language', 'tmmluplus-tve_design', 'tmmluplus-tve_mathematics', 'tmmluplus-tve_natural_sciences', 'tmmluplus-veterinary_pathology', 'tmmluplus-veterinary_pharmacology']  # list of all the subsets to use for this eval
# fmt: on


class CustomSubsetTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function="tmmluplus_harness",
            hf_repo="MediaTek-Research/TCEval-v2",
            metric=["loglikelihood_acc"],
            hf_avail_splits=["test", "dev"],
            evaluation_splits=["test"],
            few_shots_split="dev",
            few_shots_select="sequential",
            suite=["community", "tc-eval-v2"],
            generation_size=1,
            stop_sequence=["\n"],
            output_regex=None,
            frozen=False,
        )


def tmmluplus_harness(line, task_name: str = None):
    topic = line["subject"]
    query = f"以下是關於{topic.replace('_', ' ')}的多選題（附答案）。\n\n"
    query += line["question"] + "\n"
    choices = [
        line['A'],
        line['B'],
        line['C'],
        line['D'],
    ]
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, choices)])
    query += "答案:"

    gold_ix = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    "__few_shots" in line and line["__few_shots"] is True  # We are adding few shots

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        instruction=f"以下是關於{topic.replace('_', ' ')}的多選題（附答案）。\n\n",
        target_for_fewshot_sorting=[" A", " B", " C", " D"][gold_ix],
    )



# STORE YOUR EVALS
SUBSET_TASKS = [CustomSubsetTask(name=f"tc-eval-v2:{subset}", hf_subset=subset) for subset in SAMPLE_SUBSETS]
# _TASKS = SUBSET_TASKS + [task]
_TASKS = SUBSET_TASKS 


# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
