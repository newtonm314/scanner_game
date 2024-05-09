from gymnasium.envs.registration import register

register(
    id="scanner_game/ScanWorld-v0",
    entry_point="scanner_game.envs:ScanWorldEnv",
)
