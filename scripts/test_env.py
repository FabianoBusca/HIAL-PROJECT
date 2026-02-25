from time import sleep
from envs.task_envs import PnPNewRobotEnv
from utils.env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper


def main():
    env = PnPNewRobotEnv(render=True)
    env = ActionNormalizer(env)
    env = ResetWrapper(env)
    env = TimeLimitWrapper(env, max_steps=150)
    env.reset(seed=0)

    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step {step+1} success={info.get('is_success', False)}")
        sleep(0.01)
        step += 1

    env.close()


if __name__ == "__main__":
    main()