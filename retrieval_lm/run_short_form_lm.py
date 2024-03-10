#!/usr/bin/python
# -*- coding: UTF-8 -*-

from transformers import AutoTokenizer, RobertaTokenizerFast, AutoConfig, RobertaForSequenceClassification
from vllm import LLM, SamplingParams
import random
import torch
import numpy as np
from tqdm import tqdm
import json
import argparse
from tqdm import tqdm
import time
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens, LM_PROMPT_DICT
from metrics import match, accuracy


seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

lm_device = 'cuda:1'


def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer


def call_model_rerank_w_scores_batch(prompt, evidences, model, max_new_tokens=15,
                                     ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                     use_seqscore=False, threshold=0.5,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False, lm_model=None, lm_tokenizer=None, raw_prompt=None):
    results = {}
    if mode != "always_retrieve":
        inputs = lm_tokenizer(raw_prompt, return_tensors="pt",
                              padding=True, truncation=True, max_length=512).to(lm_device)

        outputs = lm_model(**inputs)

        probs = torch.softmax(outputs.logits[0], dim=-1)

    # save relevance token scores
    if mode == "always_retrieve":
        do_retrieve = True

    elif mode == "no_retrieval":
        do_retrieve = False

    else:
        if threshold is not None:
            do_retrieve = (probs[1] / (
                probs[1] + probs[0]) > threshold).item()

        else:
            do_retrieve = (torch.argmax(probs).item() == 1).item()
        do_retrieve = True

    if do_retrieve is True:

        evidence_augmented_inputs = [raw_prompt + "\nEvidence: {0}\n{1}".format(
            para["title"], para["text"]) for para in evidences]

        inputs = lm_tokenizer(evidence_augmented_inputs, return_tensors="pt",
                              padding=True, truncation=True, max_length=512).to(lm_device)
        outputs = lm_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        # relevance_score_dict = {}
        # grd_score_dict = {}
        # ut_score_dict = {}
        overall_scores = {}

        ut_scores_normalized = [-1, -0.5, 0, 0.5, 1]
        for p_idx, prob in enumerate(probs):

            relevance_scores_lst = prob[3:5].tolist()
            grd_scores_lst = prob[10:13].tolist()
            ut_scores_lst = prob[5:10].tolist()

            relevance_score = relevance_scores_lst[1] / \
                sum(relevance_scores_lst)

            ground_score = (grd_scores_lst[0] / sum(grd_scores_lst)) + 0.5 * (
                grd_scores_lst[1] / sum(grd_scores_lst[:2]))

            ut_sum = sum(ut_scores_lst)
            utility_score = sum(
                score * (prob / ut_sum) for score, prob in zip(ut_scores_normalized, ut_scores_lst))


            final_score = w_rel * relevance_score + \
                w_sup * ground_score + w_use * utility_score

            overall_scores[p_idx] = {"final_score": final_score,
                                     "relevance_score": relevance_score,
                                     "ground_score": ground_score,
                                     "utility_score": utility_score,
                                     #  "relevance_score_dict": relevance_score_dict,
                                     #  "grd_score_dict": grd_score_dict,
                                     #  "ut_score_dict": utility_score
                                     }
            results[p_idx] = final_score

    else:
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
        prompt += "[No Retrieval]"
        preds = model.generate([prompt], sampling_params)

        pred = preds[0].outputs[0].text

    # Aggregating answers
    if not do_retrieve:
        postprocessed_pred = postprocess_answer_option_conditioned(pred)
        return postprocessed_pred, results, do_retrieve
    else:
        answer2score = {}
        # if closed is True:
        #     for key, result in results.items():
        #         if key == "no_retrieval":
        #             continue
        #         answer = postprocess_answer_option_conditioned(result["pred"])
        #         score = result["score"]
        #         answer2score.setdefault(answer, 0)
        #         answer2score[answer] += score
        #     sorted_answers = sorted(
        #         answer2score.items(), key=lambda x: x[1], reverse=True)
        #     best_option = sorted_answers[0][0]
        # else:
        if True:
            best_path = sorted(results.items(),
                               key=lambda x: x[1], reverse=True)[0][0]

            sampling_params = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=5000)
            preds = model.generate([prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
                evidences[best_path]["title"], evidences[best_path]["text"])], sampling_params)

            best_option = preds[0].outputs[0].text
        return best_option, results, do_retrieve


def process_data_evidences(demonstration, top_n):
    ctx_key = "ctxs" if "ctxs" in demonstration else "top_contexts"
    prompt = PROMPT_DICT["prompt_no_input"].format_map(demonstration)
    evidences = demonstration[ctx_key][:top_n]
    return prompt, evidences


def preprocess_input_data(dataset, task=None):
    new_data = []
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None

    for item in dataset:
        if task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = instruction + \
                "\n\n### Input:\n" + item["question"] + choices
            item["answers"] = [item["answerKey"]]
        else:
            prompt = instruction + "\n\n## Input:\n\n" + \
                item["question"] if instruction is not None else item["question"]
            item["instruction"] = prompt
        new_data.append(item)

    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default=".cache")
    parser.add_argument("--ndocs", type=int, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--dtype",  type=str, default="half",
                        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
    # Decoding hyperparams
    parser.add_argument('--threshold', type=float,
                        default=None, help="Adaptive threshold.")
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--use_groundness", action="store_true",
                        help="use ground score")
    parser.add_argument(
        "--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width",  type=int,
                        default=2, help="beam search width")
    parser.add_argument("--max_depth",  type=int,
                        default=2, help="tree depth width")
    parser.add_argument("--w_rel",  type=float, default=1.0,
                        help="reward weight for document relevance")
    parser.add_argument("--w_sup",  type=float, default=1.0,
                        help="reward weight for generation support (attribution)")
    parser.add_argument("--w_use",  type=float, default=1.0,
                        help="reward weight for overall completeness / utility.")
    parser.add_argument('--mode', type=str, help="mode to control retrieval.",
                        default="default", choices=['adaptive_retrieval', 'no_retrieval', 'always_retrieve'],)
    parser.add_argument('--metric', type=str,
                        help="metric to be used during evaluation")
    args = parser.parse_args()
    gpt = args.model_name
    input_path = args.input_file
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    input_data = preprocess_input_data(
        input_data, task=args.task)

    print(len(input_data))

    model_id = 'roberta-base'

    id2label = {i: label for i, label in enumerate(["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]", "[Irrelevant]", "[Relevant]",
                                                    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]", "[Fully supported]", "[Partially supported]", "[No support / Contradictory]"])}
    config = AutoConfig.from_pretrained(model_id)
    config.update({"id2label": id2label})

    lm_tokenizer = RobertaTokenizerFast.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(gpt, padding_side="left")

    model = LLM(model=gpt, download_dir=args.download_dir,
                dtype=args.dtype, tensor_parallel_size=args.world_size,)
    lm_model = RobertaForSequenceClassification.from_pretrained(
        '../data_creation/roberta_model/checkpoint-13200', config=config)
    lm_model.to(lm_device)
    lm_model.eval()

    # Get token ids for reflection tokens.
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility)

    def generate(prompt, evidences, max_new_tokens, raw_prompt):
        return call_model_rerank_w_scores_batch(prompt, evidences=evidences, model=model, max_new_tokens=max_new_tokens,
                                                rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                threshold=args.threshold, use_seqscore=args.use_seqscore,
                                                w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use, mode=args.mode, closed=args.task in ["fever", "arc_c"], lm_model=lm_model, lm_tokenizer=lm_tokenizer, raw_prompt=raw_prompt)

    preds = []
    prompts = []
    golds = []
    metric_results = []
    scores = []
    all_results = []
    count = 0
    start = time.time()
    for i, row in tqdm(enumerate(input_data)):
        results = {}
        prompt = PROMPT_DICT["prompt_no_input"].format_map(row)
        raw_prompt = LM_PROMPT_DICT["prompt_no_input"].format_map(row)

        _, evidences = process_data_evidences(row, top_n=args.ndocs)

        pred, results, do_retrieve = generate(
            prompt, evidences, max_new_tokens=args.max_new_tokens, raw_prompt=raw_prompt)

        if type(pred) is str and pred[0] == "#" or pred[0] == ":":
            pred = pred[1:]
        prompts.append(prompt)
        preds.append(pred)
        all_results.append(results)
        if do_retrieve is True:
            count += 1
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if type(
                row["answer"]) is str else row["answer"]
        if args.metric == "accuracy":
            metric_result = accuracy(pred, row["output"])

        elif args.metric == "match":
            if "SUPPORTS" in pred:
                pred = "true"
            elif "REFUTES" in pred:
                pred = "false"
            metric_result = match(pred, row["answers"])
        else:
            raise NotImplementedError

        metric_results.append(metric_result)
        if i % 10 == 0:
            print("average: {}".format(np.mean(metric_results)))
            # final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
            #                  "golds": golds,  "metric":  args.metric, "metric_mean": np.mean(metric_results), "scores": scores}
            # with open(args.output_file + "_tmp", "w") as outfile:
            #     json.dump(final_results, outfile)

    # final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
    #                  "golds": golds,  "metric":  args.metric, "metric_mean": np.mean(metric_results), "scores": scores}
    # with open(args.output_file, "w") as outfile:
    #     json.dump(final_results, outfile)

    print("Final result: {0}".format(np.mean(metric_results)))
    print("Retrieval Frequencies: {0}".format(count / len(input_data)))
    print('TIME:', time.time() - start)


if __name__ == "__main__":
    main()


# python3 run_short_form_lm.py --model_name selfrag/selfrag_llama2_7b --input_file ../eval_data/popqa_longtail_w_gs.jsonl --mode adaptive_retrieval --max_new_tokens 100 --threshold 0.2 --output_file temp --metric match --ndocs 10 --use_groundness --use_utility --use_seqscore --dtype half
# python3 run_short_form_lm.py --model_name selfrag/selfrag_llama2_7b --input_file ../eval_data/triviaqa_test_w_gs.jsonl --mode adaptive_retrieval --max_new_tokens 100 --threshold 0.2 --output_file temp --metric match --ndocs 10 --use_groundness --use_utility --use_seqscore --dtype half
# python3 run_short_form_lm.py --model_name selfrag/selfrag_llama2_7b --input_file ../eval_data/health_claims_processed.jsonl --mode adaptive_retrieval --max_new_tokens 50 --threshold 0.2 --output_file temp --metric match --ndocs 5 --use_groundness --use_utility --use_seqscore --dtype half --task fever
# python3 run_short_form_lm.py --model_name selfrag/selfrag_llama2_7b --input_file ../eval_data/arc_challenge_processed.jsonl --mode adaptive_retrieval --max_new_tokens 50 --threshold 0.2 --output_file temp --metric match --ndocs 5 --use_groundness --use_utility --use_seqscore --dtype half --task arc_c
