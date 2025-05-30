import random
import sys
import os

import openai
import time

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
        except openai.RateLimitError:
            sleep_len += 1
            print(f"Rate limit error. Increasing sleep length to {sleep_len}s.")
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

    instruction = "Instruction: Explore the reasoning behind all the labels. Then, select the label that has the most valid reasoning."

    assert os.path.basename(args.train_file) in {"esnli_no.txt", "anli_v1.txt", "qqp_no.txt"}
    assert os.path.basename(args.reasoning_file) in {"nli_fewshot.txt", "nli_fewshot.txt"}
    example_prompt = load_prompt(args.train_file)
    reasoning_prompt = load_prompt(args.reasoning_file)

    new_token = 100
    example_list = example_prompt.split("\n###")
    label_set = meta_dict["QQP"].values() if args.task in ["QQP", "PAWS"] else meta_dict["ESNLI"].values()

    outfile_name = args.train_file.replace("_no", "").replace(".txt", "_all_reason.txt")

    output_list = []
    sleep_len = args.init_sleep_len
    with open(outfile_name, "w") as outfile:
        for example in example_list:
            if example == "":
                continue
            question, answer = example.strip().split("\nLabel: ")
            prompt_no_label = reasoning_prompt + "\n" + question
            reason_list = []
            for c in label_set:
                prompt_reason_c = "{}\nLabel: {}".format(prompt_no_label, c)
                response, sleep_len = backoff_response(
                    client, prompt_reason_c, sleep_len, args.model_name, args.temp, new_token
                )
                print(response)
                response = response.strip().split("###")[0].strip()
                reason_flag = False
                for line in response.split("\n"):
                    line = line.strip()
                    if line.startswith("Reason:"):
                        reason_list.append("Possible Reasoning for {}: {}".format(c, line.replace("Reason: ", "")))
                        reason_flag = True
                        break
                if reason_flag is False:
                    reason_list.append("Possible Reasoning for {}: None".format(c))
            output = "{}\n{}\n{}\nLabel: {}\n".format(question, instruction, "\n".join(reason_list), answer)
            output_list.append(output)
        outfile.write("###\n".join(output_list))


if __name__ == "__main__":
    parser = args()
    args = parser.parse_args()
    print(" ".join(sys.argv))
    print(args)

    main(args)

    print("\nDone")
