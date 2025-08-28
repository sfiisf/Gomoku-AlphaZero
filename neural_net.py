import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class NeuralNet(nn.Module):
    def __init__(self, game, in_channels=3, hidden_channels=32, num_residual=5):
        super(NeuralNet, self).__init__()

        self.board_size = game.get_board_size()
        self.num_actions = self.board_size * self.board_size

        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(hidden_channels)

        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_residual)
        ])

        # 策略头
        self.policy_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_size * self.board_size, self.num_actions)

        # 价值头
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_size * self.board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        # 策略
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        # policy = F.log_softmax(policy, dim=1)

        # 价值
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 3:
            x = x.unsqueeze()
        device = next(self.parameters()).device
        x = x.to(device)

        policy_logits, value = self.forward(x)
        policy = torch.softmax(policy_logits, dim=1)

        H = W = self.board_size
        policy_2d = policy[0].view(H, W)
        return policy_2d