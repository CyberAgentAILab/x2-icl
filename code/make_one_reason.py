import random
import sys
import os

from data_utils import *
from utils_api import *


def main(args):
    random.seed(args.seed)

    reason_template = "Possible Reasoning for "
    label_template = "\nLabel: "

    assert "_all_reason" in os.path.basename(args.train_file)
    example_prompt = load_prompt(args.train_file)

    example_list = example_prompt.split("\n###")

    outfile_name = args.train_file.replace("_all_reason", "_one_reason")
    output_list = []
    with open(outfile_name, "w") as outfile:
        for example in example_list:
            if example == "":
                continue
            question, answer = example.strip().split(label_template)
            question_decomposed = question.split("\n")
            for line in question_decomposed:
                line = line.replace(reason_template, "")
                if line.startswith(answer):
                    reason_for_correct = line.replace("{}: ".format(answer), "").strip()
                    break
            if "qqp_" in args.train_file:
                qa = (
                    question_decomposed[:3]
                    + ["Reason: {}".format(reason_for_correct)]
                    + ["Label: {}\n".format(answer)]
                )
            else:
                qa = (
                    question_decomposed[:2]
                    + ["Reason: {}".format(reason_for_correct)]
                    + ["Label: {}\n".format(answer)]
                )
            qa = "\n".join(qa)
            output_list.append(qa)
        outfile.write("###\n".join(output_list))


if __name__ == "__main__":
    parser = args()
    args = parser.parse_args()
    print(" ".join(sys.argv))
    print(args)

    main(args)

    print("\nDone")
