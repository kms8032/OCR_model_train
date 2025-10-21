import torch
import torch.nn as nn
import os

# ==== 기본 설정 ====
IMG_W, IMG_H = 128, 32
CHARS = "0123456789"

class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden=256, rnn_layers=2, rnn_dropout=0.3):
        super().__init__()
        def conv_bn_relu(i_c, o_c, k=3, s=1, p=1, use_bn=True, drop_p=0.0):
            layers = [nn.Conv2d(i_c, o_c, k, s, p, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(o_c))
            layers.append(nn.ReLU(inplace=True))
            if drop_p > 0:
                layers.append(nn.Dropout2d(drop_p))
            return nn.Sequential(*layers)

        self.cnn = nn.Sequential(
            conv_bn_relu(1,   64, 3,1,1), nn.MaxPool2d(2,2),
            conv_bn_relu(64, 128, 3,1,1), nn.MaxPool2d(2,2),
            conv_bn_relu(128,256,3,1,1),
            conv_bn_relu(256,256,3,1,1), nn.MaxPool2d(2,2),
            conv_bn_relu(256,512,3,1,1), conv_bn_relu(512,512,3,1,1), nn.MaxPool2d(2,2),
            nn.Conv2d(512,512,(2,1),1,0), nn.ReLU(inplace=True)
        )

        self.rnn = nn.LSTM(
            input_size=512, hidden_size=hidden,
            num_layers=rnn_layers, bidirectional=True,
            batch_first=False, dropout=rnn_dropout if rnn_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden*2, num_classes)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.squeeze(2)
        feat = feat.permute(2,0,1)
        seq, _ = self.rnn(feat)
        logits = self.fc(seq)
        return logits

# ==== 변환 ====
def convert_pth_to_onnx(pth_path="checkpoints/crnn_best.pth", onnx_path="crnn_model.onnx"):
    print(f"Loading checkpoint: {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")

    num_classes = 1 + len(CHARS)
    model = CRNN(num_classes)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy_input = torch.randn(1, 1, IMG_H, IMG_W)
    dynamic_axes = {"input": {0: "batch"}, "output": {1: "batch"}}

    print("Exporting to ONNX...")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=13,
        do_constant_folding=True
    )
    print(f"Exported ONNX file: {onnx_path}")

if __name__ == "__main__":
    os.makedirs("onnx", exist_ok=True)
    convert_pth_to_onnx("checkpoints/crnn_best.pth", "onnx/crnn_model.onnx")
