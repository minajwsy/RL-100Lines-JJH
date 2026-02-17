import torch as T
import argparse

# [시스템 설정]
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

# ==============================================================================
# [환경별 하이퍼파라미터 프리셋]
# 근거: PPO 논문 (Schulman et al., 2017) 및 CleanRL 벤치마크
# - Continuous (MuJoCo, DMC, LunarLander): Table 3 파라미터 (MuJoCo Standard)
# - Discrete (CartPole): Table 4 파라미터 (Atari Standard)
# ==============================================================================
ENV_CONFIGS = {
    # --------------------------------------------------------------------------
    # 1. Gymnasium (Discrete)
    # --------------------------------------------------------------------------
    'CartPole-v1': {
        # [목표 점수] 500.0 (만점)
        'lr': 2.5e-4,           # 논문 Atari 값 (0.00025)
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.1,        # 이산 제어는 0.1이 표준
        'max_grad_norm': 0.5,
        'K_epoch': 3,           # 논문 값
        'T_horizon': 128,       # 논문 값 (짧은 호라이즌)
        'mb_size': 32,          # T_horizon / 4
        'max_timesteps': 100_000,
        'vf_coef': 0.5,         # Value Loss 가중치
        'ent_coef': 0.01        # Discrete는 탐험이 중요하므로 소량 추가
    },

    # --------------------------------------------------------------------------
    # 2. Gymnasium (Continuous)
    # --------------------------------------------------------------------------
    'LunarLanderContinuous-v3': {
        # [목표 점수] 200.0 점 이상
        'lr': 3e-4,             # 논문 값 (0.0003)
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,        # 연속 제어는 0.2가 표준
        'max_grad_norm': 0.5,
        'K_epoch': 10,          # 논문 값
        'T_horizon': 2048,      # 논문 값
        'mb_size': 64,          # 논문 값
        'max_timesteps': 1_000_000,
        'vf_coef': 0.5,
        'ent_coef': 0.0         # 쉬운 연속 제어는 엔트로피 불필요
    },

    # --------------------------------------------------------------------------
    # 3. MuJoCo (Continuous) - PPO 논문 Table 3
    # --------------------------------------------------------------------------
    'Hopper-v5': {
        # [목표 점수] 2,500 ~ 3,000 점
        'lr': 3e-4,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64,
        'max_timesteps': 1_000_000,
        'vf_coef': 0.5,
        'ent_coef': 0.0         # MuJoCo 표준은 엔트로피 0.0
    },

    'Walker2d-v5': {
        # [목표 점수] 3,000 ~ 4,000 점
        'lr': 3e-4,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64,
        'max_timesteps': 1_000_000,
        'vf_coef': 0.5,
        'ent_coef': 0.0
    },

    # --------------------------------------------------------------------------
    # 4. DeepMind Control (Continuous) - MuJoCo 설정 적용
    # --------------------------------------------------------------------------
    'dm_control/cheetah-run-v0': {
        # [목표 점수] 700 ~ 800 점 (최대 1000)
        'lr': 3e-4,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64,
        'max_timesteps': 5_000_000,
        'vf_coef': 0.5,
        'ent_coef': 0.01       # DMC는 초기 탐험이 중요하므로 0.01 권장
    },

    'dm_control/walker-walk-v0': {
        # [목표 점수] 900 점 이상 (최대 1000)
        'lr': 3e-4,
        'gamma': 0.99,
        'lmbda': 0.95,
        'eps_clip': 0.2,
        'max_grad_norm': 0.5,
        'K_epoch': 10,
        'T_horizon': 2048,
        'mb_size': 64,
        'max_timesteps': 2_000_000,
        'vf_coef': 0.5,
        'ent_coef': 0.01        # DMC Walker는 0.01 필수
    }
}

# ==============================================================================
# [사용할 환경 선택] (User Selection)
# 원하는 환경의 주석(#)을 제거하여 활성화하세요. (한 번에 하나만 선택)
# ==============================================================================

parser = argparse.ArgumentParser(description='PPO Hyperparameters')
parser.add_argument('--env', type=str, default="dm_control/cheetah-run-v0", help='Environment name')
args = parser.parse_args()
env_name = args.env
'''CartPole-v1, LunarLanderContinuous-v3, Hopper-v5, Walker2d-v5, dm_control/cheetah-run-v0, dm_control/walker-walk-v0'''

# ==============================================================================
# [자동 설정 적용] (Auto Configuration Logic)
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
    max_timesteps = cfg['max_timesteps']
    vf_coef = cfg['vf_coef']
    ent_coef = cfg['ent_coef']
else:
    # 안전 장치: 목록에 없으면 기본값 사용
    print(f"Warning: {env_name} 설정을 찾을 수 없어 MuJoCo 기본값을 사용합니다.")
    lr = 3e-4
    gamma = 0.99
    lmbda = 0.95
    eps_clip = 0.2
    K_epoch = 10
    T_horizon = 2048
    mb_size = 64
    max_grad_norm = 0.5
    max_timesteps = 1_000_000

    # [추가된 기본값]
    vf_coef = 0.5
    ent_coef = 0.0
