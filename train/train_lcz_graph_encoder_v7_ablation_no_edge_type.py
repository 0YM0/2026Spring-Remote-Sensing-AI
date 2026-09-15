#!/usr/bin/env python3
from train_lcz_graph_encoder_v7_ablation import parse_args, train


def main():
    args = parse_args()
    args.ablation = "no_edge_type"
    args.exp_name = "ablation_v7_no_edge_type"
    train(args)


if __name__ == "__main__":
    main()
