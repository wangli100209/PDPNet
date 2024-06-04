DEFAULT_DEVICE = 'cuda'

DEFAULT_BACTCHSIZE = 32

DEFAULT_PATCHSIZE = None

DEFAULT_EPOCH = 500

DEFAULT_LEARNINGRATE = 1e-3

DEFAULT_VISUALSTEP = 400

DEFAULT_SAVERULE = 'cycle_metric'

DEFAULT_EARLYSTOPTIMES = 150

DEFAULT_CYCLETIMES = 5

DEFAULT_NUMCLASSES = 2

WARMUP_FLAG = False

SAVEER_PATH = './result/demo'

CHECKPOINT_PATH = './result/demo/best_model.pth'

WUMETRIC_PATH = './result/demo/best_metrics.pkl'

TR_ROOT = './dataset/ISPY-G'

TE_ROOT = './dataset/ISPY-S'

TRAINLOGGFILE_PATH = './result/demo/trlog.csv'

TESTLOGGFILE_PATH = './result/demo/valog.csv'

INIT_METHOD = 'normal'

INIT_GAIN = 0.02