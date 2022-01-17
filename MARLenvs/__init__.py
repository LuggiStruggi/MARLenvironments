from gym.envs.registration import register

register(id='TwoStep-v0', entry_point='MARLenvs.two_step_env:TwoStepEnv')
register(id='Switch-v0', entry_point='MARLenvs.switch_env:SwitchEnv')
register(id='Navigation-v0', entry_point='MARLenvs.navigation_env:NavigationEnv')
