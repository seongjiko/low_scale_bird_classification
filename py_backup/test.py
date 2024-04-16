from datasets import *
from models import *
from trainer import *
import random, os
import json

def seed_everything(seed=42):
    random.seed(seed)  # Python 내장 random 모듈
    os.environ['PYTHONHASHSEED'] = str(seed)  # 환경변수 설정
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU 시드 고정
    torch.cuda.manual_seed(seed)  # PyTorch GPU 시드 고정
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경에서도 시드 고정
    torch.backends.cudnn.deterministic = True  # CuDNN 관련 설정
    torch.backends.cudnn.benchmark = False  # 동일한 입력 크기의 데이터가 반복될 경우 속도 향상을 위한 벤치마크 모드 비활성화


def test_start(cfg, seed, pt_name = ''):
    seed_everything(seed=seed)

    model = get_deit3_large()
    state_dict = torch.load(f'models/{cfg["attempt_name"]}.pt')
    model.load_state_dict(state_dict)
    print(f"test model(pretrained) load 완료!")

    test_loader = get_loader(cfg, is_test=True)
    print(f"data load 완료!")

    print('추론을 진행합니다.')
    preds = run_model(model, test_loader, None, None, is_training=False, epoch=cfg['epochs'], is_test=True)

    return preds


