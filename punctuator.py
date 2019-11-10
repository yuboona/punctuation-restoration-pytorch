import train.train_use_conf as train
import inference.inference as inference
from conf import TRAINCONF as args

# Training Option
train.parser.add_argument(
    '--train', '-tr', action='store_true',
    help='training'
)
# Test Option
train.parser.add_argument(
    '--test', '-t', action='store_true',
    help='training'
)


if __name__ == "__main__":
    s = train.parser.parse_args()
    if s.train is True:
        print("Start Training\n", "*"*80)
        train.main(args)
    elif s.test is True:
        print("Inferencing", "*"*80)
        inference.main(args)
