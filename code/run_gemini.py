# Author   : Xuanli He
# Version  : 1.0
# Filename : run.py, calculate_acc.py
# Modified and extended by: Ukyo Honda
import random
import sys
import os

import google.generativeai as genai
import time

from tqdm import tqdm
from collections import defaultdict
from data_utils import *
from utils_api import *


def get_response(
    model,
    prompt,
    temp,
    max_tokens,
):
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temp,
            ),
        )
        return response.text
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "None"


def backoff_response(
    model,
    prompt,
    temp,
    max_tokens,
):
    response = None
    while response is None:
        try:
            response = get_response(model, prompt, temp, max_tokens)
        except Exception as e:
            print(f"Error: {e}")
    return response


def main(args):
    genai.configure(api_key=str(os.getenv("GEMINI_API_KEY")))
    model_path = model_libs[args.model_name]

    label_template = "Label: "

    model = genai.GenerativeModel(model_path)

    random.seed(args.seed)

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

    base_prompt = load_prompt(args.train_file)

    if "all_reason" in args.train_file:
        new_token = 600
    elif "one_reason" in args.train_file:
        new_token = 300
    else:
        new_token = 100

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

            pred_list = []
            response_list = []
            correct_flag = False
            for _ in range(args.sample):
                response = backoff_response(model, prompt, args.temp, new_token)
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
    args.outfile = os.path.join(out_dir, outfile)

    main(args)

    print("\nDone")
