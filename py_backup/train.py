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


def train_start(cfg, seed, pretrained=False, pretrained_model_pt=''):
    seed_everything(seed=seed)

    filename = 'method_log.json'
    # 파일이 존재하는지 확인하고, 존재하면 기존 내용을 읽습니다.
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:  # 파일이 비어있거나 JSON 형식이 아닌 경우
                data = {}
    else:
        data = {}

    # 'attempt_name'을 키로 하여 'method' 값을 갱신하거나 추가합니다.
    data[cfg['attempt_name']] = cfg['method']

    # 변경된 데이터를 파일에 씁니다.
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"'{cfg['attempt_name']}' 설정이 '{filename}' 파일에 저장되었습니다.")

    model = get_deit3_large()
    if pretrained:
        state_dict = torch.load(f'models/{pretrained_model_pt}.pt')
        model.load_state_dict(state_dict)
    print(f"model load 완료!")
    
    train_loader, valid_loader = get_loader(cfg)
    print(f"data load 완료!")

    if cfg['use_kfold']:
        print('Kfold 학습 모드입니다. 학습을 진행합니다.')
        train_kfold(model, cfg, train_loader, valid_loader)
        
    else:
        print('학습을 진행합니다.')
        train(model, cfg, train_loader, valid_loader)

    return model


