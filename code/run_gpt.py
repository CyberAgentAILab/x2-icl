# Author   : Xuanli He
# Version  : 1.0
# Filename : run.py, calculate_acc.py
# Modified and extended by: Ukyo Honda
import random
import sys
import os

import openai
import time

from tqdm import tqdm
from collections import defaultdict
from data_utils import *
from utils_api import *


def get_response(
    client,
    prompt,
    model,
    temp,
    max_tokens,
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
        max_tokens=max_tokens,
        frequency_penalty=0,
        presence_penalty=0,
    )
    if response.choices[0].finish_reason == "content_filter":
        print(f"ContentFilterError: {response.choices[0].content_filter_results}")
        return "ContentFilterError"
    return response.choices[0].message.content


def backoff_response(
    client,
    prompt,
    sleep_len,
    model,
    temp,
    max_tokens,
):
    response = None
    while response is None:
        try:
            time.sleep(sleep_len)
            response = get_response(client, prompt, model, temp, max_tokens)
        except openai.RateLimitError as e:
            sleep_len += 1
            print(f"Rate limit error: {e}\nIncreasing sleep length to {sleep_len}s.")
        except Exception as e:
            print(f"Unexpected error: {e}")
    return response, sleep_len


def main(args):
    endpoint = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
    api_key = str(os.getenv("AZURE_OPENAI_API_KEY"))
    client = openai.AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=args.api_version,
    )

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

    label_template = "Label: "

    base_prompt = load_prompt(args.train_file)

    new_token = 500

    prompts, answers = read_data(base_prompt, args.test_file, data_loaders[args.task], test_examples=500, binary=False)
    if args.debug:
        prompts = prompts[:10]
        answers = answers[:10]
    assert len(prompts) == len(answers)

    start_time = time.time()
    correct_dict = defaultdict(list)
    sleep_len = args.init_sleep_len
    with open(args.outfile, "w") as outfile:
        print("\nPROMPT\n{}".format(base_prompt))
        outfile.write("PROMPT\n{}\n".format(base_prompt))
        for i, (prompt, answer) in tqdm(enumerate(zip(prompts, answers)), desc="Instances", total=len(prompts)):
            if args.train_file.endswith("_all_reason.txt"):
                inst = "Instruction: Explore the reasoning behind all the labels. Then, select the label that has the most valid reasoning."
                prompt = prompt + "\n" + inst
            if args.train_file.endswith("_one_reason_inst.txt"):
                inst = "Instruction: Explain the reasoning and then select the correct label from entailment, neutral, or contradiction."
                prompt = prompt + "\n" + inst
            if args.add_zcot:
                inst = "Let's think step by step. Ensure that your response ends with Label: and your final answer."
                prompt = prompt + "\n" + inst
            if args.add_zcot_all:
                inst = "Let's think step by step, exploring the reasons why each label could be correct. Ensure that your response ends with Label: and your final answer."
                prompt = prompt + "\n" + inst

            pred_list = []
            response_list = []
            correct_flag = False
            for _ in range(args.sample):
                response, sleep_len = backoff_response(
                    client, prompt, sleep_len, args.model_name, args.temp, new_token
                )
                response = response.strip().split("###")[0].strip()
                pred = None
                for line in response.split("\n"):
                    line = line.strip()
                    if line.startswith(label_template):
                        pred = line.replace(label_template, "")
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
        print("\n\nAll: Total {}, Acc {}".format(i + 1, sum([sum(v) for v in correct_dict.values()]) / (i + 1) * 100))
        outfile.write(
            "\n\nAll: Total {}, Acc {}\n".format(i + 1, sum([sum(v) for v in correct_dict.values()]) / (i + 1) * 100)
        )
        for k, v in correct_dict.items():
            print("{}: Total {}, Acc {}".format(k, len(v), sum(v) / len(v) * 100))
            outfile.write("{}: Total {}, Acc {}\n".format(k, len(v), sum(v) / len(v) * 100))
        end_time = time.time()
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
    if args.add_zcot:
        assert args.train_file in {"anli_v1.txt", "qqp_no.txt", "esnli_no.txt"}
        outfile = "zcot_" + outfile
    if args.add_zcot_all:
        assert args.train_file in {"anli_v1.txt", "qqp_no.txt", "esnli_no.txt"}
        outfile = "zcotall_" + outfile
    args.outfile = os.path.join(out_dir, outfile)

    main(args)

    print("\nDone")
