# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch.optim import AdamW
from deepspeed.ops.adam import FusedAdam
from .salmonn import SALMONN
import deepspeed

def load_model(model_config, run_config):
    # model = SALMONN.from_config(config)
    # model = deepspeed.initialize(model=model, config_params=config.deepspeed_config)[0]  # DeepSpeed로 래핑
    # return model
    model = SALMONN.from_config(model_config)

    # 옵티마이저 생성
    optimizer = FusedAdam(
        model.parameters(),
        lr=run_config.optims.init_lr,
        betas=(0.9, run_config.optims.beta2),
        weight_decay=run_config.optims.weight_decay
    )

    # DeepSpeed 초기화
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config_params=model_config.deepspeed_config
    )
    return model_engine
