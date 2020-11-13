import argparse

from trainer import Trainer


def main(args):
    durations = [0, 1 / 32, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 3 / 16, 1 / 4, 1 / 3, 3 / 8, 1 / 2, 2 / 3, 3 / 4, 1]
    if args.mode == 'TRAIN':
        trainer = Trainer(model_name=args.model, style=args.style, resume=())
        trainer.train()
    elif args.mode == 'PREDICT':
        trainer = Trainer(model_name=args.model, style=args.style, resume=(args.src, args.epoch))
        trainer.predict(args.model, args.style, 1024, durations, start_index=0)
    elif args.mode == 'ORI':
        trainer = Trainer(model_name=args.model, style=args.style, resume=())
        trainer.originality()
    elif args.mode == 'GEN':
        for i in range(10):
            trainer = Trainer(model_name=args.model, style=args.style, resume=(args.src, i))
            trainer.predict(args.model, args.style, 1024, durations, start_index=i * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['VAE', 'Transformer'], required=True)
    parser.add_argument('--style', type=str, choices=['CSQ', 'CPI'], required=True)
    parser.add_argument('--mode', type=str, choices=['TRAIN', 'PREDICT', 'ORI', 'GEN'], required=True)
    parser.add_argument('--src', type=str)
    parser.add_argument('--epoch', type=int)
    arguments = parser.parse_args()
    main(arguments)
