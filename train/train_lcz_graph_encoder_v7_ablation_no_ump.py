#!/usr/bin/env python3
from train_lcz_graph_encoder_v7_ablation import parse_args, train


def main():
    args = parse_args()
    args.ablation = "no_ump_tokens"
    args.exp_name = "ablation_v7_no_ump"
    train(args)


if __name__ == "__main__":
    main()
