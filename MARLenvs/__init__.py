from gym.envs.registration import register

register(id='FirstEnv-v0', entry_point='MARLenvs.first_env:FirstEnv')
register(id='TwoStepEnv-v0', entry_point='MARLenvs.two_step_env:TwoStepEnv')
