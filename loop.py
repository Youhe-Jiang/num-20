# Owner(s): ["oncall: distributed"]

import sys
import torch
import torch.nn as nn
from torch.nn import Linear, Module, Sequential
from torch import distributed as dist
import torchvision
import numpy as np
from torch.distributed._fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear
from torch.optim import SGD
from torch.optim import Adam
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    get_full_params,
)
from torch.testing._internal.common_utils import TEST_WITH_DEV_DBG_ASAN, run_tests
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import time
def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


class TestUnevenParamShard(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_one_iteration(self):
        """Test FSDP with parameter shards."""
        print_peak_memory("Peak Memory before loading model", self.rank)
        my_lr = 0.1
        num = 600
        model_1 = FSDP(nn.Sequential(
                *[nn.Linear(num, num) for _ in range(24)]
                #nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1),
        ))
        model_2 = nn.Sequential(
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(12)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(12)])),
        )
        model_3 = nn.Sequential(
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(8)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(8)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(8)])),
        )
        model_4 = nn.Sequential(
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(6)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(6)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(6)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(6)])),
        )
        model_5 = nn.Sequential(
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(4)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(4)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(4)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(4)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(4)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(4)])),
        )
        model_6 = nn.Sequential(
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(3)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(3)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(3)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(3)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(3)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(3)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(3)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(3)])),
        )
        model_7 = nn.Sequential(
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(2)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(2)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(2)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(2)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(2)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(2)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(2)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(2)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(2)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(2)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(2)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(2)])),
        )
        model_8 = nn.Sequential(
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
                FSDP(nn.Sequential(*[nn.Linear(num, num) for _ in range(1)])),
        )
        model = sys.argv[1]
        model = FSDP(model.cuda())
        print_peak_memory("Load model to GPU...", self.rank)
        optim = SGD(model.parameters(), lr=my_lr)
        '''
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('../num-20/model_8_test/600'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            for i in range(10):
            #data = torch.randn(1, 1024, 3, 3)
            #target = torch.randn(1, 1024, 3, 3)
                data = torch.randn(1, num)
                target = torch.randn(1, num)
                output = model(data.cuda())
                print_peak_memory("Output result from model...", self.rank)
                loss = F.cross_entropy(output, target.cuda())
                print_peak_memory("Peak Memory before optimizer.step()", self.rank)
                optim.step()
                print_peak_memory("Peak Memory after optimizer.step()", self.rank)
                optim.zero_grad()
                prof.step()
        '''
        for i in range(10):
            #data = torch.randn(1, 1024, 3, 3)
            #target = torch.randn(1, 1024, 3, 3)
                data = torch.randn(1, num)
                target = torch.randn(1, num)
                output = model(data.cuda())
                print_peak_memory("Output result from model...", self.rank)
                loss = F.cross_entropy(output, target.cuda())
                print_peak_memory("Peak Memory before optimizer.step()", self.rank)
                optim.step()
                print_peak_memory("Peak Memory after optimizer.step()", self.rank)
                optim.zero_grad()
if __name__ == "__main__":
    run_tests()
