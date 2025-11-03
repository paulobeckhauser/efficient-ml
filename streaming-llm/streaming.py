import argparse

def main(args):
    model_name_or_path = args.model_name_or_path
    print(f"Model: {model_name_or_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3")
    args = parser.parse_args()
    main(args)
