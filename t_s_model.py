"""
Date: 2019/11/20 16:21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

KD_loss = nn.KLDivLoss(reduction='batchmean')


def kd_step(teacher: nn.Module,
            student: nn.Module,
            temperature: float,
            inputs: torch.tensor,
            optimizer):
    teacher.eval()  # 不更新参数
    student.train() # 更新参数

    with torch.no_grad():
        logits_t = teacher(inputs=inputs)

    logits_s = student(inputs=inputs)

    loss = KD_loss(input=F.log_softmax(logits_s/temperature, dim=-1),
                   target=F.softmax(logits_t/temperature, dim=-1))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
