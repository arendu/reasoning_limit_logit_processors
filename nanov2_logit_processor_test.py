# copied from here https://docs.vllm.ai/en/v0.10.1.1/examples/offline_inference/logits_processor.html
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""This example demonstrates instantiating vLLM with a custom logits processor
class object.

For a basic example of implementing a custom logits processor, see
the `DummyLogitsProcessor` implementation in `vllm/test_utils.py`.

For testing purposes, a dummy logits processor is employed which, if
`target_token` is passed as a keyword argument to `SamplingParams.extra_args`,
will mask out all tokens except `target_token`.

A batch is constructed with `temperature=0.0` and 50% of requests specifying
`target_token`, and for these requests - and *only* these requests - we
expect the `target_token` to be decoded in each step, yielding an output
similar to that shown below:

Generated Outputs:
------------------------------------------------------------
Prompt:    'Hello, my name is'
Output:    " ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' '"
------------------------------------------------------------
Prompt:    'The president of the United States is'
Output:    " not a racist. He is a racist.\nHe's a racist because he"
------------------------------------------------------------
Prompt:    'The capital of France is'
Output:    ' also also also also also also also also also also also also also
             also also also'
------------------------------------------------------------
Prompt:    'The future of AI is'
Output:    ' in the hands of the people.\n\nThe future of AI is in the'
------------------------------------------------------------
"""

from types import DynamicClassAttribute
from typing import Optional, Dict, Any, List

import torch

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)
#import os
#os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

NEWLINE_TOKENS = {12291, 8199, 20487, 57354, 71693, 14350, 126989, 2064, 10260, 26644, 114710, 10263, 106525, 81952, 116768, 10274, 51235, 28710, 94246, 79914, 86059, 2092, 79917, 18479, 18481, 96306, 32819, 2100, 14390, 67639, 73782, 94265, 110650, 122935, 2116, 12357, 114758, 86087, 67656, 4169, 100423, 104519, 106570, 114759, 124997, 102479, 12368, 14416, 14417, 38997, 98389, 34903, 67673, 102490, 26720, 94306, 12390, 61546, 53355, 32876, 98414, 26740, 90228, 94325, 39031, 127095, 18554, 47229, 6273, 51330, 4227, 2180, 98437, 100485, 112779, 96397, 108686, 43151, 4241, 110737, 69780, 49301, 86173, 4256, 118944, 18595, 16548, 102565, 12455, 114859, 92334, 104623, 6320, 118960, 118963, 55476, 67766, 14519, 55479, 123067, 61631, 32960, 2241, 63685, 114885, 116935, 63691, 125131, 57551, 104655, 73942, 100571, 16606, 73954, 16616, 114925, 14576, 51440, 20723, 4342, 67830, 14588, 123134, 33024, 4353, 39170, 26885, 67846, 123142, 129286, 55561, 28938, 37131, 84236, 12560, 88337, 43282, 6422, 24866, 90404, 14630, 45352, 20777, 31020, 20783, 14642, 65846, 8503, 4410, 100670, 14657, 53569, 2373, 20806, 90445, 6478, 104785, 41301, 20822, 115029, 8538, 6491, 47456, 94561, 24931, 72038, 117098, 31086, 43374, 43378, 80242, 14708, 61813, 82293, 53623, 92535, 110964, 39293, 94592, 104833, 72068, 72071, 10632, 84360, 98696, 55692, 22926, 27022, 31120, 35215, 117138, 74136, 123293, 37278, 86430, 106910, 37281, 125346, 113060, 68007, 65960, 88489, 98729, 43438, 33203, 39347, 102839, 57784, 49593, 76218, 108983, 68030, 92606, 16832, 100798, 14790, 43462, 37321, 4554, 86475, 96716, 14797, 43472, 29141, 12758, 2519, 109013, 14810, 16858, 72156, 10717, 61918, 125407, 80352, 84451, 12776, 76264, 92650, 6637, 80365, 125422, 8691, 104947, 92665, 23034, 14843, 12795, 14845, 72189, 14847, 107007, 66050, 64006, 4615, 12807, 74247, 80390, 123401, 14863, 25104, 90641, 35346, 37395, 2580, 94736, 84503, 2591, 102946, 76323, 74283, 18988, 70189, 121388, 125483, 14896, 98865, 100915, 90676, 19002, 35390, 80447, 47680, 98879, 86596, 100932, 27207, 2634, 43597, 2638, 107087, 2640, 4688, 121426, 82515, 113236, 8789, 21078, 8791, 88661, 127578, 21090, 82531, 19044, 84578, 94822, 33383, 96867, 115307, 25197, 62062, 64112, 10869, 72310, 8824, 14971, 17020, 43646, 2687, 78464, 66177, 12930, 27266, 86659, 96894, 119428, 129666, 8841, 123531, 29324, 103055, 2706, 113300, 111257, 31391, 82592, 37541, 15014, 49833, 2731, 96940, 68269, 2734, 115377, 94903, 21181, 62142, 31423, 86720, 6849, 43715, 45763, 72388, 76485, 92867, 94919, 17100, 51919, 49872, 60113, 58067, 27350, 117462, 41690, 10971, 53979, 56028, 10974, 31454, 53982, 115419, 39653, 78566, 127727, 31473, 35571, 60147, 64244, 35574, 97015, 119546, 2812, 31484, 84733, 127743, 45825, 2820, 21253, 78597, 27399, 29448, 11017, 97033, 6923, 27404, 35596, 86796, 101126, 117520, 84754, 2838, 35608, 8985, 76569, 82713, 125721, 70433, 49958, 29479, 82729, 2861, 60206, 37679, 66357, 113462, 2871, 74551, 119606, 19261, 13118, 78657, 97091, 74566, 113483, 6991, 47952, 35669, 11095, 35674, 105306, 95068, 13149, 2913, 35683, 9060, 11108, 31588, 66404, 23400, 107366, 123751, 33645, 127854, 58226, 9075, 47988, 25461, 21366, 80758, 19323, 62332, 107387, 115580, 31624, 103304, 25482, 54158, 68494, 50068, 93078, 11159, 13207, 80791, 115607, 56219, 97181, 117662, 21407, 41890, 33699, 84898, 86947, 129960, 78768, 25525, 19383, 19384, 60343, 74681, 119736, 58303, 113599, 19394, 95172, 5062, 119750, 109514, 60363, 15308, 95179, 43982, 121809, 76754, 9171, 23507, 62421, 80855, 23513, 70623, 76772, 74726, 27624, 100335, 111595, 1010, 7154, 60403, 80888, 44025, 29690, 97278, 7168, 11270, 5127, 62473, 58380, 54285, 76816, 107537, 107539, 3092, 56343, 39960, 44059, 3100, 115739, 19486, 44065, 101411, 31782, 89128, 23593, 105512, 109611, 11308, 125994, 68655, 117811, 87093, 121909, 7223, 9273, 44090, 7227, 66620, 107577, 130110, 3135, 11330, 9283, 80962, 97347, 99397, 23626, 25676, 95309, 123980, 29775, 115792, 50259, 68692, 87126, 27736, 42076, 19551, 19553, 99425, 58467, 103527, 66666, 50283, 95338, 19565, 99437, 44143, 130154, 99442, 117876, 40058, 50298, 111739, 11389, 11390, 7295, 68735, 7297, 87168, 66691, 97412, 23685, 58502, 126084, 78984, 19594, 23691, 76944, 19602, 11411, 25748, 101522, 109715, 130199, 130200, 40089, 52377, 115869, 19614, 17572, 9381, 76968, 107688, 79020, 33966, 13487, 50352, 111791, 58547, 93363, 91318, 54455, 52408, 56504, 5306, 76984, 83135, 70848, 25799, 11471, 38097, 21714, 60625, 40151, 79063, 91352, 5338, 111832, 77021, 46302, 9439, 50400, 3297, 3298, 25828, 52454, 23784, 36080, 107760, 1267, 3318, 17655, 118010, 15611, 70909, 111870, 120061, 97536, 60673, 87298, 93441, 115976, 124171, 7437, 7438, 48403, 5396, 34069, 58643, 64790, 66835, 109848, 46366, 52511, 21794, 60707, 21796, 44325, 83235, 93475, 105766, 23849, 101673, 46380, 27949, 19758, 103726, 68912, 56626, 19767, 40247, 1338, 5439, 73024, 87362, 25923, 54597, 7498, 1355, 116042, 9549, 9551, 5457, 56657, 70993, 81235, 1365, 34133, 19802, 34138, 52572, 32093, 109916, 120155, 89445, 73064, 11625, 56683, 111979, 109940, 21877, 3448, 56697, 71032, 103804, 64895, 71040, 111999, 60802, 73091, 48516, 62853, 17798, 73093, 7562, 32140, 109964, 85394, 36243, 118171, 7580, 13724, 13725, 21916, 60830, 71069, 71071, 114080, 118177, 3493, 128422, 21927, 73127, 3500, 1456, 21942, 120246, 1468, 36285, 32194, 3523, 36297, 50634, 19920, 15829, 93653, 30168, 128472, 3546, 17882, 69082, 114140, 101854, 1512, 11753, 9706, 3563, 13803, 46570, 60907, 1520, 36336, 22004, 69108, 124407, 48632, 95738, 52733, 85501, 75265, 30211, 99843, 24071, 48647, 87559, 122375, 32268, 128535, 1561, 48668, 11810, 32291, 1572, 83494, 38439, 95783, 114219, 89644, 85551, 120367, 101940, 75319, 3640, 52797, 1600, 91712, 28230, 3655, 20039, 22087, 34378, 38471, 46665, 65102, 75342, 56913, 38482, 116306, 120408, 1626, 34395, 7772, 73307, 83549, 30303, 91740, 54881, 28258, 120413, 13926, 83558, 1640, 1641, 104042, 106092, 11886, 28272, 56950, 65142, 44664, 24185, 99959, 106108, 128638, 59009, 11906, 26244, 65159, 24202, 56977, 34450, 61074, 24212, 102038, 93852, 81566, 18081, 46753, 79523, 50852, 9893, 69285, 85665, 104104, 61100, 7854, 28335, 128687, 61105, 102066, 26293, 24246, 38582, 3768, 61112, 57018, 81592, 93884, 22205, 126654, 124608, 89793, 38594, 114377, 28364, 130764, 20174, 63183, 14032, 50898, 9940, 28378, 46811, 102109, 57054, 124643, 83684, 44775, 55015, 3817, 79594, 48875, 67307, 3824, 110322, 57075, 67316, 9973, 16117, 20213, 12024, 55031, 120568, 69371, 32508, 100092, 104190, 130808, 65280, 38659, 12039, 1801, 28426, 77579, 53004, 61198, 32530, 108306, 59158, 67350, 3864, 30489, 130842, 102173, 5920, 16161, 1826, 69411, 87840, 87842, 16166, 24362, 3885, 18225, 128818, 32563, 1844, 89907, 53047, 71482, 108346, 44861, 69437, 71488, 69441, 89922, 94017, 100167, 32584, 1877, 110423, 124760, 106331, 53084, 63325, 42846, 65374, 100191, 20321, 38754, 106341, 118631, 100202, 116588, 112493, 14190, 36719, 46958, 71537, 102254, 89974, 104312, 12156, 38781, 10110, 3971, 71556, 40838, 112518, 65416, 36745, 57226, 87946, 10127, 92048, 3989, 34710, 75670, 118680, 20377, 24474, 71578, 38816, 1953, 71586, 24483, 6052, 83877, 110496, 128930, 67499, 53165, 28590, 122797, 20400, 126895, 102323, 30645, 10166, 63413, 92089, 114617, 49083, 57277, 120765, 55234, 20419, 40900, 79811, 88002, 38858, 4043, 38859, 65483, 57294, 83917, 104396, 2002, 26579, 30679, 120791, 88025, 57311, 26592, 30689, 12260, 100324, 53223, 71657, 2030, 4078, 14320, 22512, 28656, 8179, 4084, 32753, 67568, 79855, 61432, 122875, 116734}

class ThinkingBudgetLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        # Store a mapping from request index to output_tok_ids reference
        self.logit_processor_state: dict[int, dict[Any, Any]] = {}

    def is_argmax_invariant(self) -> bool:
        return False  # This processor does not affect sampling

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return
        # Add new requests
        for index, sampling_params, prompt_tok_ids, output_tok_ids in batch_update.added:
            state = self.logit_processor_state.get(index, {})
            state["output_tok_ids"] = output_tok_ids
            state["thinking_budget"] = sampling_params.extra_args["thinking_budget"]
            state["thinking_budget_grace_period"] = sampling_params.extra_args["thinking_budget_grace_period"]
            state["end_token_ids"] = sampling_params.extra_args["end_token_ids"]
            state["is_thinking"] = False
            print(f" end of prompt tokens are {prompt_tok_ids[-3:]}")
            if prompt_tok_ids[-3:] == [49250, 2077, 1561]:  # check for \n<think>\n at the end of the prompt which indicates that the model is in thinking mode.
                state["is_thinking"] = True
                state["start_of_end"] = False
                state["end_of_end"] = False
            self.logit_processor_state[index] = state
        # Remove finished requests
        for index in batch_update.removed:
            self.logit_processor_state.pop(index, None)
        # Handle moved requests
        for a, b, direction in batch_update.moved:
            a_val = self.logit_processor_state.pop(a, None)
            b_val = self.logit_processor_state.pop(b, None)
            if a_val is not None:
                self.logit_processor_state[b] = a_val
            if direction.name == "SWAP" and b_val is not None:
                self.logit_processor_state[a] = b_val
    
    def _suffix_prefix_overlap(self, a, b):
        m = min(len(a), len(b))
        for k in range(m, 0, -1):           # try longest first
            if a[-k:] == b[:k]:
                return k
        return 0
    
    def _maybe_end_thinking(self, idx: int, logits: torch.Tensor, state: Dict[Any, Any]):
        if state["end_of_end"]:
            return logits
        
        if len(state["output_tok_ids"]) >= state["thinking_budget"] + state["thinking_budget_grace_period"] and not state["start_of_end"]:
            state["start_of_end"] = True

        if len(state["output_tok_ids"]) >= state["thinking_budget"] and state["output_tok_ids"][-1] in NEWLINE_TOKENS and not state["start_of_end"]:
            state["start_of_end"] = True
        
        if state["start_of_end"] and not state["end_of_end"]:
            end_token_ids = state["end_token_ids"]
            last_n_inputs = list(state["output_tok_ids"][-len(end_token_ids):])
            overlap = self._suffix_prefix_overlap(last_n_inputs, end_token_ids)
            if overlap < len(end_token_ids):
                logits[idx, :] = float("-inf") 
                insert_id = end_token_ids[overlap]
                logits[idx, insert_id] = 1.0
            else:
                state["end_of_end"] = True
        return logits

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for idx, state in self.logit_processor_state.items():
            if state.get("is_thinking", False):
                logits = self._maybe_end_thinking(idx, logits, state)
        return logits

model="nvidia/NVIDIA-Nemotron-Nano-12B-v2"
# Sample prompts.

msg = """Bob is an avid fan of the video game \"League of Leesins\", and today he celebrates as the League of Leesins World Championship comes to an end! \n\nThe tournament consisted of $n$ ($n \\ge 5$) teams around the world. Before the tournament starts, Bob has made a prediction of the rankings of each team, from $1$-st to $n$-th. After the final, he compared the prediction with the actual result and found out that the $i$-th team according to his prediction ended up at the $p_i$-th position ($1 \\le p_i \\le n$, all $p_i$ are unique). In other words, $p$ is a permutation of $1, 2, \\dots, n$.\n\nAs Bob's favorite League player is the famous \"3ga\", he decided to write down every $3$ consecutive elements of the permutation $p$. Formally, Bob created an array $q$ of $n-2$ triples, where $q_i = (p_i, p_{i+1}, p_{i+2})$ for each $1 \\le i \\le n-2$. Bob was very proud of his array, so he showed it to his friend Alice.\n\nAfter learning of Bob's array, Alice declared that she could retrieve the permutation $p$ even if Bob rearranges the elements of $q$ and the elements within each triple. Of course, Bob did not believe in such magic, so he did just the same as above to see Alice's respond.\n\nFor example, if $n = 5$ and $p = [1, 4, 2, 3, 5]$, then the original array $q$ will be $[(1, 4, 2), (4, 2, 3), (2, 3, 5)]$. Bob can then rearrange the numbers within each triple and the positions of the triples to get $[(4, 3, 2), (2, 3, 5), (4, 1, 2)]$. Note that $[(1, 4, 2), (4, 2, 2), (3, 3, 5)]$ is not a valid rearrangement of $q$, as Bob is not allowed to swap numbers belong to different triples.\n\nAs Alice's friend, you know for sure that Alice was just trying to show off, so you decided to save her some face by giving her any permutation $p$ that is consistent with the array $q$ she was given. \n\n\n-----Input-----\n\nThe first line contains a single integer $n$ ($5 \\le n \\le 10^5$) — the size of permutation $p$.\n\nThe $i$-th of the next $n-2$ lines contains $3$ integers $q_{i, 1}$, $q_{i, 2}$, $q_{i, 3}$ ($1 \\le q_{i, j} \\le n$) — the elements of the $i$-th triple of the rearranged (shuffled) array $q_i$, in random order. Remember, that the numbers within each triple can be rearranged and also the positions of the triples can be rearranged.\n\nIt is guaranteed that there is at least one permutation $p$ that is consistent with the input. \n\n\n-----Output-----\n\nPrint $n$ distinct integers $p_1, p_2, \\ldots, p_n$ ($1 \\le p_i \\le n$) such that $p$ is consistent with array $q$. \n\nIf there are multiple answers, print any. \n\n\n-----Example-----\nInput\n5\n4 3 2\n2 3 5\n4 1 2\n\nOutput\n1 4 2 3 5\n\n\nRead the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```"""
messages = [
    {"role": "system", "content": "You are a helpful assistant. /think"},
    {"role": "user", "content": msg}

]
messages2= [
    {"role": "system", "content": "You are a helpful assistant. /think"},
    {"role": "user", "content": "Write a haiku about a cat"}

]
tokenizer = AutoTokenizer.from_pretrained(model)
prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False),
           tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True, add_special_tokens=False)]
# Create a mixture of requests which do and don't utilize the dummy logitproc
sampling_params_list = [
    SamplingParams(temperature=0.6, max_tokens=220, extra_args={"thinking_budget": 50, "thinking_budget_grace_period": 0, "end_token_ids":[1871, 5565, 11483, 6139, 1046, 2259, 74045, 1062]}),
    SamplingParams(temperature=0.6, max_tokens=260, extra_args={"thinking_budget": 20, "thinking_budget_grace_period": 0, "end_token_ids":[2259, 74045, 1062]}),
]

def main():
    # Create an LLM.
    llm = LLM(
        model=model,
        logits_processors=[ThinkingBudgetLogitsProcessor],
        trust_remote_code=True,
    )
    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params_list)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()