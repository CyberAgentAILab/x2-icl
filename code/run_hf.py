# Author   : Xuanli He
# Version  : 1.0
# Filename : run.py, calculate_acc.py
# Modified and extended by: Ukyo Honda
import random
import sys
import os

import torch
import time

from tqdm import tqdm
from collections import defaultdict
from data_utils import *
from utils_hf import *


def main(args):
    model_path = model_libs[args.model_name]
    if args.dtype == "bf16":
        args.dtype = torch.bfloat16
    elif args.dtype == "fp16":
        args.dtype = torch.float16
    elif args.dtype == "fp32":
        args.dtype = torch.float32
    else:
        raise ValueError("Invalid dtype")
    tokenizer, model = create_model(model_path, "cuda", True, args.dtype, args.quantize)

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    data_loaders = {
        "HANS": HANS,
        "NAN": NAN,
        "ISCS": ISCS,
        "ST": ISCS,
        "LILI": ISCS,
        "PICD": ISCS,
        "PISP": ISCS,
        "ESNLI": ESNLI,
        "ANLI": ANLI,
        "QQP": QQP,
        "PAWS": QQP,
    }

    label_template = "Label: "

    base_prompt = load_prompt(args.train_file)

    new_token = 500
    if args.model_name.startswith("deepseek-r1"):
        new_token = 1500

    prompts, answers = read_data(base_prompt, args.test_file, data_loaders[args.task], test_examples=500, binary=False)
    if args.debug:
        prompts = prompts[:10]
        answers = answers[:10]
    assert len(prompts) == len(answers)

    label_set = meta_dict["QQP"].values() if args.task in ["QQP", "PAWS"] else meta_dict["ESNLI"].values()

    start_time = time.time()
    correct_dict = defaultdict(list)
    with open(args.outfile, "w") as outfile:
        print("\nPROMPT\n{}".format(base_prompt))
        outfile.write("PROMPT\n{}\n".format(base_prompt))
        for i, (prompt, answer) in tqdm(enumerate(zip(prompts, answers)), desc="Instances", total=len(prompts)):
            if args.train_file.endswith("_all_reason.txt"):
                inst = "Instruction: Explore the reasoning behind all the labels. Then, select the label that has the most valid reasoning."
                prompt = prompt + "\n" + inst
            message = [
                {
                    "role": "system",
                    "content": "Answer the question by following the provided examples. Ensure that your response ends with {}and your final answer.".format(
                        label_template
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            # DeepSeek-R1's chat_template was changed after our experiments.
            # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/commit/74fbf131a939963dd1e244389bb61ad0d0440a4d
            # To reproduce our experiments, remove <think> here and let the model start generating from <think>.
            if args.model_name.startswith("deepseek-r1"):
                prompt = prompt.replace("<think>\n", "")

            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            max_len = input_ids.input_ids.shape[-1] + new_token

            pred_list = []
            response_list = []
            correct_flag = False
            for _ in range(args.sample):
                with torch.inference_mode():
                    if args.sample == 1:
                        generate_ids = model.generate(
                            **input_ids, do_sample=False, top_p=None, top_k=None, temperature=None, max_length=max_len
                        )
                    else:
                        generate_ids = model.generate(
                            **input_ids, do_sample=True, top_p=0.9, top_k=None, temperature=0.7, max_length=max_len
                        )
                response = tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                prompt_len = len(
                    tokenizer.batch_decode(
                        input_ids.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                )
                response = response[prompt_len:]
                response = response.strip()
                pred = None
                for line in response.split("\n"):
                    line = line.replace("**", "").replace("###", "").strip()
                    if line.startswith(label_template):
                        _pred = line.replace(label_template, "").strip().split()[0]
                        if _pred in label_set:
                            pred = _pred
                            break
                if pred is not None:
                    pred_list.append(pred)
                response_list.append(response.replace("\n", " "))
            if len(pred_list) == 0:
                pred = None
            else:
                pred = max(set(pred_list), key=pred_list.count)
            if pred == answer:
                correct_dict[answer].append(1)
                correct_flag = True
            elif answer == "not entailment" and pred in ["contradiction", "neutral"]:
                correct_dict[answer].append(1)
                correct_flag = True
            else:
                correct_dict[answer].append(0)

            q = prompt.split("###\n")[-1].strip()
            qa = "{} Label: {}".format(q.replace("\n", " "), answer)
            print("\n{}. {}\nPred: {} -> {}, OX: {}".format(i, qa, pred_list, pred, "O" if correct_flag else "X"))
            outfile.write(
                "\n{}. {}\nPred: {} -> {}, OX: {}\n".format(i, qa, pred_list, pred, "O" if correct_flag else "X")
            )
            for j, r in enumerate(response_list):
                print("Sample {}: {}".format(j, r))
                outfile.write("Sample {}: {}\n".format(j, r))
        assert i + 1 == sum([len(v) for v in correct_dict.values()])
        end_time = time.time()
        print("\n\nAll: Total {}, Acc {}".format(i + 1, sum([sum(v) for v in correct_dict.values()]) / (i + 1) * 100))
        outfile.write(
            "\n\nAll: Total {}, Acc {}\n".format(i + 1, sum([sum(v) for v in correct_dict.values()]) / (i + 1) * 100)
        )
        for k, v in correct_dict.items():
            print("{}: Total {}, Acc {}".format(k, len(v), sum(v) / len(v) * 100))
            outfile.write("{}: Total {}, Acc {}\n".format(k, len(v), sum(v) / len(v) * 100))
        print(
            "Time: {:.2f} sec, {:.4f} min, {:.6f} hour".format(
                end_time - start_time, (end_time - start_time) / 60, (end_time - start_time) / 3600
            )
        )
        outfile.write(
            "Time: {:.2f} sec, {:.4f} min, {:.6f} hour".format(
                end_time - start_time, (end_time - start_time) / 60, (end_time - start_time) / 3600
            )
        )
        outfile.write("\n{}\n{}".format(" ".join(sys.argv), args))


if __name__ == "__main__":
    parser = args()
    args = parser.parse_args()
    print(" ".join(sys.argv))
    print(args)

    if args.task == "ANLI":
        test_file = os.path.basename(args.test_file)
        test_file = test_file.replace(".jsonl", "")
        out_dir = "output/{}/{}/".format(test_file, args.model_name)
    else:
        out_dir = "output/{}/{}/".format(args.task, args.model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dirs, pname = os.path.split(args.train_file)
    pname = pname.replace(".txt", "")
    _, dir_seed = os.path.split(dirs)
    dir_seed = dir_seed.replace("seed", "")
    assert int(dir_seed) == args.seed
    outfile = "{}_{}_n{}_t{}.txt".format(pname, dir_seed, args.sample, args.temp)
    if args.quantize != "full":
        outfile = "{}_{}".format(args.quantize, outfile)
    args.outfile = os.path.join(out_dir, outfile)

    main(args)

    print("\nDone")
