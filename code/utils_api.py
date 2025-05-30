# Author   : Xuanli He
# Version  : 1.0
# Filename : utils.py
# Modified and extended by: Ukyo Honda
import argparse

model_libs = {
    "gpt-4o-2024-08-06": "gpt-4o-2024-08-06",
    "gemini-1.5-pro-002": "gemini-1.5-pro-002",
    "gemini-2.0-flash-001": "gemini-2.0-flash-001",
}

data_libs = {
    "HANS": "HANS",
    "NAN": "NAN",
    "ISCS": "ISCS",
    "ST": "ISCS",
    "LILI": "ISCS",
    "PICD": "ISCS",
    "PISP": "ISCS",
    "ESNLI": "ESNLI",
    "ANLI": "ANLI",
    "QQP": "QQP",
    "PAWS": "QQP",
}


def args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, choices=list(model_libs.keys()))

    parser.add_argument(
        "--train_file",
        type=str,
        help="In-context demonstration examples",
        required=True,
    )

    parser.add_argument(
        "--test_file",
        type=str,
        help="test file for the evaluation",
        required=True,
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=list(data_libs.keys()),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="random seed",
    )

    parser.add_argument("--temp", type=float, default=0.0, help="temperature")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--api_version", type=str, default="2024-09-01-preview")
    parser.add_argument("--init_sleep_len", type=float, default=0.0, help="amount of time between requests")
    parser.add_argument("--sample", type=int, default=1, help="number of samples to generate")

    parser.add_argument("--add_zcot", default=False, action="store_true")
    parser.add_argument("--add_zcot_all", default=False, action="store_true")

    return parser
