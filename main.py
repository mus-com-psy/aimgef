import argparse
from models.trainer import Trainer
from utilities.preprocess import preprocess
import utilities.crawler as crawler


def main(args):
    if args.style == "CSQ":
        time_set = [0, 1 / 32, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 3 / 16, 1 / 4, 1 / 3, 3 / 8, 1 / 2, 2 / 3, 3 / 4, 1]
    else:
        time_set = False

    if args.task == 'TRAIN':
        trainer = Trainer(model_name=args.model, style=args.style, resume=None)
        trainer.train()
    elif args.task == 'PREDICT':
        trainer = Trainer(model_name=args.model, style=args.style, resume=(args.src, args.epoch))
        trainer.predict(args.model, args.style, 1024, time_set, start_index=0)
    elif args.task == 'ORI':
        trainer = Trainer(model_name=args.model, style=args.style, resume=())
        trainer.originality()
    elif args.task == 'GEN':
        # for i in ['1-100', '1-300', '1-1000', '1-5000'] + list(range(1, 31)):
        trainer = Trainer(model_name=args.model, style=args.style, resume=(args.src, args.epoch))
        trainer.predict(args.model, args.style, 1024, time_set, start_index=0)
            # trainer.predict(args.model, args.style, 1024, time_set, start_index=50)
    elif args.task == 'CSQ_DATA':
        # crawler.main()
        preprocess(style='CSQ', aug=True)
    elif args.task == 'CPI_DATA':
        preprocess(style='CPI', representation='token', aug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['VAE', 'Transformer'])
    parser.add_argument('--style', type=str, choices=['CSQ', 'CPI'])
    parser.add_argument('--task', type=str, choices=['TRAIN', 'PREDICT', 'ORI', 'GEN', 'CPI_DATA', 'CSQ_DATA'])
    parser.add_argument('--src', type=str)
    parser.add_argument('--epoch', type=int)
    arguments = parser.parse_args()
    main(arguments)
