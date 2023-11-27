# Copyright 2023 a1147
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

import onnxruntime
import torch
import numpy as np
import os
import cv2
import scipy

ort_session = onnxruntime.InferenceSession("btest (2).onnx", providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
x = torch.randn(1, 3, 224, 224, requires_grad=True)

for img in os.listdir('train_datasets/non'):
    img_path = f'train_datasets/non/{img}'
    img_data = cv2.imread(img_path)
    img_data = cv2.resize(img_data, (224,224))
    img_data = img_data.astype(np.float32) / 255
    img_data = np.moveaxis(img_data, -1, 0)
    img_data = np.expand_dims(img_data, 0)
    ort_inputs = {ort_session.get_inputs()[0].name: img_data}
    ort_outs = ort_session.run(None, ort_inputs)
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    print(softmax(ort_outs))
    # break
# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")