import argparse
from model.trainer import Trainer
from model.preprocess import preprocess
import model.utilities.crawler as crawler


def main(args):
    if args.style == "CSQ":
        durations = [0, 1 / 32, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 3 / 16, 1 / 4, 1 / 3, 3 / 8, 1 / 2, 2 / 3, 3 / 4, 1]
    else:
        durations = False
    if args.mode == 'TRAIN':
        trainer = Trainer(model_name=args.model, style=args.style, resume=None)
        trainer.train()
    elif args.mode == 'PREDICT':
        trainer = Trainer(model_name=args.model, style=args.style, resume=(args.src, args.epoch))
        trainer.predict(args.model, args.style, 1024, durations, start_index=0)
    elif args.mode == 'ORI':
        trainer = Trainer(model_name=args.model, style=args.style, resume=())
        trainer.originality()
    elif args.mode == 'GEN':
        # for i in ['1-100', '1-300', '1-1000', '1-5000'] + list(range(1, 31)):
        trainer = Trainer(model_name=args.model, style=args.style, resume=(args.src, args.epoch))
        trainer.predict(args.model, args.style, 1024, durations, start_index=0)
            # trainer.predict(args.model, args.style, 1024, durations, start_index=50)
    elif args.mode == 'CSQ_DATA':
        crawler.main()
        preprocess(style='CSQ', representation='token', aug=True)
    elif args.mode == 'CPI_DATA':
        preprocess(style='CPI', representation='token', aug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['VAE', 'Transformer'])
    parser.add_argument('--style', type=str, choices=['CSQ', 'CPI'])
    parser.add_argument('--mode', type=str, choices=['TRAIN', 'PREDICT', 'ORI', 'GEN', 'CPI_DATA', 'CSQ_DATA'])
    parser.add_argument('--src', type=str)
    parser.add_argument('--epoch', type=int)
    arguments = parser.parse_args()
    main(arguments)
