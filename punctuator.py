import train.train_use_conf as train

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

s = train.parser.parse_args()

if s.train is True:
    print(train.args)
    train.main(train.args)
