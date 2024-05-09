import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class ScanWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, field_size=5, scan_radius=1, max_steps=200):
        self.field_size = field_size  # The size of the square grid
        self.scan_radius = scan_radius
        self.window_size = 512  # The size of the PyGame window
        self.max_steps = max_steps

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, field_size - 1, shape=(2,), dtype=int),
                "field": spaces.MultiBinary((field_size,field_size)),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "field": self._field_discovered}

    def _get_info(self):
        return {
            "episode_reward": self.total_reward
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.field_size, size=2, dtype=int)

        self._field_discovered = np.zeros((self.field_size,self.field_size),dtype='int8')

        self.steps_count = 0
        self.total_reward = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.field_size - 1
        )

        old_field = np.copy(self._field_discovered)

        r_lb = max(0, self._agent_location[0]-self.scan_radius)
        r_ub = min(self.field_size-1, self._agent_location[0]+self.scan_radius)
        c_lb = max(0, self._agent_location[1]-self.scan_radius)
        c_ub = min(self.field_size-1, self._agent_location[1]+self.scan_radius)
        for i in range(r_lb, r_ub+1):
            for j in range(c_lb, c_ub+1):
                self._field_discovered[i][j] = 1


        self.steps_count += 1

        terminated = self._field_discovered.all()
        trunctated = self.steps_count >= self.max_steps

        if terminated:
            reward = 1
        else:
            reward = (np.sum(self._field_discovered) - np.sum(old_field) - 1)/self.field_size**2

        self.total_reward += reward
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, trunctated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.field_size
        )  # The size of a single grid square in pixels

        # Draw discovered region
        for i in range(self.field_size):
            for j in range(self.field_size):
                if self._field_discovered[i][j]:
                    pygame.draw.rect(
                        canvas,
                        (127, 127, 255),
                        pygame.Rect(
                            pix_square_size * np.array([i,j]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Draw active scan region
        r_lb = max(0, self._agent_location[0]-self.scan_radius)
        r_ub = min(self.field_size-1, self._agent_location[0]+self.scan_radius)
        c_lb = max(0, self._agent_location[1]-self.scan_radius)
        c_ub = min(self.field_size-1, self._agent_location[1]+self.scan_radius)
        for i in range(r_lb, r_ub+1):
            for j in range(c_lb, c_ub+1):
                pygame.draw.rect(
                        canvas,
                        (127, 255, 127,),
                        pygame.Rect(
                            pix_square_size * np.array([i,j]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.field_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__=='__main__':
    try:
        env = ScanWorldEnv(render_mode='human', field_size=5, scan_radius=1)
   
        while True:
            obs, info = env.reset()
            done_episode = False
            while not done_episode:
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                obs = next_obs
                done_episode = terminated or truncated
    except KeyboardInterrupt:
        pass