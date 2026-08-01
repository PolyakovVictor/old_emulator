import random
import numpy as np
import tinygrad.nn as nn

from PIL import Image
from pyboy import PyBoy
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.nn.state import get_parameters
from collections import deque
from pyboy.utils import WindowEvent


Tensor.training = True


MAX_BUFF = 1000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.00025


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
    for press_evt in ACTIONS[action_idx]:
        game.send_input(press_evt)


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


class MarioEnv:
    def __init__(self, rom_path='game_mario.gb'):
        self.game = PyBoy(rom_path)
        self.game.set_emulation_speed(0) # no speed limit
        self.prior_x = self.get_mario_x()

    def get_mario_x(self,):
        page = self.game.memory[0xC0AB]
        offset = self.game.memory[0xC202]
        return page * 256 + offset
    
    def is_dead(self,):
        return self.game.memory[0xC0AC] == 0x06

    def reset(self): self.prior_x = self.get_mario_x()
    
    def step(self, action_idx, frame_skip=4):
        apply_action(self.game, action_idx)

        for _ in range(frame_skip):
            self.game.tick()
        
        current_x = self.get_mario_x()
        
        reward = float(current_x - self.prior_x)
        self.prior_x = current_x
        

        done = self.is_dead()
        if done:
            reward = -100.00
        
        reward -= 0.1

        return reward, done

 

class DQN:
    def __init__(self) -> None:
        self.conv1 = nn.Conv2d(4,16,kernel_size=5,stride=2)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=2)
        
        self.fc1 = nn.Linear(32 * 16 * 18, 256)
        self.fc2 = nn.Linear(256, 5)
    
    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = x.flatten(1)
        x = self.fc1(x).relu()
        return self.fc2(x)


def preprocess_frame(raw_frame):
    rgb = raw_frame[:, :, :3]
    gray = np.dot(rgb, [0.299, 0.587, 0.144]).astype(np.uint8)
    downsampled = gray[::2, ::2]
    return downsampled


mario = MarioEnv()
game = mario.game

policy_net = DQN()
optimizer = nn.optim.Adam(get_parameters(policy_net), lr=LR)

buffer = ReplayBuffer(MAX_BUFF)
frame_stack = deque(maxlen=4)


initial_frame = preprocess_frame(game.screen.ndarray)

for _ in range(4):
    frame_stack.append(initial_frame)


epsilon = 1.0
EPS_MIN = 0.1
EPS_DECAY = 0.9995

step = 0
while step < 10000:
    step += 1
    state = np.array(frame_stack) # 4,72,80
    

    if random.random() < epsilon:
        action = random.randint(0,4)
    else:
        state_tensor = Tensor(state, dtype='float32').unsqueeze(0) / 255.0
        q_values = policy_net(state_tensor).realize()
        action = int(q_values.argmax(axis=1).numpy()[0])
    epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
    
    reward, done = mario.step(action)

    next_frame = preprocess_frame(game.screen.ndarray)
    frame_stack.append(next_frame)
    next_state = np.array(frame_stack)


    buffer.push(state, action, reward, next_state, done)
    if done: mario.reset()
    if len(buffer) >= BATCH_SIZE and step % 4 == 0:
        b_s, b_a, b_r, b_ns, b_d = buffer.sample(BATCH_SIZE)
        q_all = policy_net(b_s)
        q_values = (q_all * b_a.one_hot(5)).sum(axis=1)
        next_q_all = policy_net(b_s).detach()
        max_next_q = next_q_all.max(axis=1)
        target_q = b_r + (GAMMA * max_next_q * (1.0 - b_d))
        loss = ((q_values - target_q) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"Step: {step} | Loss: {loss.numpy():.4f} | Epsilon: {epsilon:.3f}")
    # print(step)

game.stop()
