import random
import numpy as np


from PIL import Image
from pyboy import PyBoy
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from collections import deque
from pyboy.utils import WindowEvent


MAX_BUFF = 1000
BATCH_SIZE = 32


ACTIONS = [
    [], # Do nothing
    [WindowEvent.PRESS_ARROW_RIGHT], # 1: Right
    [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A], # 2: Right + Jump
    [WindowEvent.PRESS_BUTTON_A], # 3: Jump
    [WindowEvent.PRESS_ARROW_LEFT],
]


RELEASE_ALL = [
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B
]


def apply_action(game: PyBoy, action_idx: int):
    for release_evt in RELEASE_ALL:
        game.send_input(release_evt)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            Tensor(np.array(state), dtype='float32') / 255.0,
            Tensor(np.array(action), dtype='int32'),
            Tensor(np.array(reward), dtype='float32'),
            Tensor(np.array(next_state), dtype='float32') / 255.0,
            Tensor(np.array(done), dtype='float32'),
        )
    
    def __len__(self,): return len(self.buffer)


def preprocess_frame(raw_frame):
    rgb = raw_frame[:, :, :3]
    gray = np.dot(rgb, [0.299, 0.587, 0.144]).astype(np.uint8)
    downsampled = gray[::2, ::2]
    return downsampled


game = PyBoy('game_mario.gb')
game.set_emulation_speed(0)

buffer = ReplayBuffer(MAX_BUFF)
frame_stack = deque(maxlen=4)


initial_frame = preprocess_frame(game.screen.ndarray)

for _ in range(4):
    frame_stack.append(initial_frame)


step = 0
while step < 10000:
    step += 1

    state = np.array(frame_stack) # 4,72,80

    action = random.randint(0,4)
    apply_action(game, action)
    
    game.tick()
    next_frame = preprocess_frame(game.screen.ndarray)
    frame_stack.append(next_frame)
    next_state = np.array(frame_stack)
    
    reward = 0.0
    done = False
    
    buffer.push(state, action, reward, next_state, done)
    if len(buffer) >= BATCH_SIZE and step % 10 == 0:
        b_s, b_a, b_r, b_ns, b_d = buffer.sample(BATCH_SIZE)
    # print(step)

game.stop()
