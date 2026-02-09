import torch as T

# [시스템 설정]
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

# ==============================================================================
# [환경별 하이퍼파라미터 프리셋]
# 목표 점수(Target Score)는 작성하신 PPO 모델(Orthogonal Init + LR Annealing) 기준입니다.
# ==============================================================================
ENV_CONFIGS = {
    # 1. Gymnasium (Discrete): 막대 균형 잡기
    # [Target Score] 500.0 (만점 유지 필수)
    'CartPole-v1': {
        'lr': 0.0005,           # 간단한 환경이라 LR을 조금 높게 잡음
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.1,        # Discrete는 clip을 작게 잡는 게 안정적일 때가 많음
        'max_grad_norm': 0.5,
        'K_epoch': 4,           # 많이 학습할 필요 없음
        'T_horizon': 128,       # 짧은 호라이즌으로 빠른 피드백
        'mb_size': 32
    },

    # 2. Gymnasium (Continuous): 달 착륙선
    # [Target Score] 200.0 이상 (Solved 기준)
    'LunarLanderContinuous-v3': {
        'lr': 0.0003,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64
    },

    # 3. MuJoCo: 외발 점프 로봇 (균형 감각 테스트)
    # [Target Score] 3000.0 이상
    'Hopper-v5': {
        'lr': 0.0003,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64
    },

    # 4. MuJoCo: 이족 보행 로봇 (관절 제어 심화)
    # [Target Score] 3500.0 이상
    'Walker2d-v5': {
        'lr': 0.0003,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64
    },

    # 5. DeepMind Control: 치타 달리기 (Gym과 다름)
    # [Target Score] 700 ~ 800점 (Max 1000)
    'dm_control/cheetah-run-v0': {
        'lr': 0.0003,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64
    },

    # 6. DeepMind Control: 워커 걷기
    # [Target Score] 900점 이상 (Max 1000)
    'dm_control/walker-walk-v0': {
        'lr': 0.0003,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64
    }
}

# ==============================================================================
# [사용할 환경 선택] (User Selection)
# 원하는 환경의 주석(#)을 제거하여 활성화하세요. (한 번에 하나만 선택)
# ==============================================================================

# [Gym Discrete]
# env_name = 'CartPole-v1'

# [Gym Continuous]
# env_name = 'LunarLanderContinuous-v3'

# [MuJoCo]
env_name = 'Hopper-v5'
# env_name = 'Walker2d-v5'

# [DeepMind Control]
# env_name = 'dm_control/cheetah-run-v0'
# env_name = 'dm_control/walker-walk-v0'


# ==============================================================================
# [자동 설정 적용] (Auto Configuration Logic)
# 선택된 env_name에 맞춰 파라미터를 자동으로 변수에 할당합니다.
# ==============================================================================
if env_name in ENV_CONFIGS:
    cfg = ENV_CONFIGS[env_name]

    lr = cfg['lr']
    gamma = cfg['gamma']
    lmbda = cfg['lmbda']
    eps_clip = cfg['eps_clip']
    K_epoch = cfg['K_epoch']
    T_horizon = cfg['T_horizon']
    mb_size = cfg['mb_size']
    max_grad_norm = cfg['max_grad_norm']
else:
    # 목록에 없는 새로운 환경일 경우 기본값 사용 (Safe Fallback)
    print(f"Warning: {env_name} 설정을 찾을 수 없어 기본값을 사용합니다.")
    lr = 3e-4
    gamma = 0.99
    lmbda = 0.95
    eps_clip = 0.2
    K_epoch = 10
    T_horizon = 2048
    mb_size = 64
    max_grad_norm = 0.5
