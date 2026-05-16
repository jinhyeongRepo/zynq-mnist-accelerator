import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# ------------------------------------------------------------------------------
# 1. 초경량 MNIST CNN 모델 정의
# ------------------------------------------------------------------------------
class LeNetLight(nn.Module):
    def __init__(self):
        super(LeNetLight, self).__init__()
        # 입력: 1x28x28
        # Zybo Z7-10 자원을 위해 채널 수를 최소화 (출력: 4x26x26)
        self.conv = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 출력: 4x13x13
        
        # 완전연결층 (4 * 13 * 13 = 676 차원 -> 10 차원)
        self.fc = nn.Linear(4 * 13 * 13, 10, bias=True)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(-1, 4 * 13 * 13)
        x = self.fc(x)
        return x

# ------------------------------------------------------------------------------
# 2. 고정소수점 양자화 알고리즘 (Quantization)
# ------------------------------------------------------------------------------
def quantize_tensor(tensor, scale):
    """
    부동소수점 텐서를 고정소수점 정수로 변환하고 반올림 및 Clamping 수행
    scale: 2^Q_FACTOR (예: Q_FACTOR가 7이면 scale은 128)
    """
    quantized = torch.round(tensor * scale).int()
    # INT8 범위(-128 ~ 127)로 제한 (필요시 INT16 변환을 위해 범위 조절 가능)
    return torch.clamp(quantized, -128, 127)

def dump_to_c_header(filename, var_name, numpy_array):
    """C언어 1차원 배열 형태의 헤더 파일로 저장"""
    flat_array = numpy_array.flatten()
    with open(filename, 'a') as f:
        f.write(f"const int8_t {var_name}[{len(flat_array)}] = {{\n    ")
        for i, val in enumerate(flat_array):
            f.write(f"{int(val)}")
            if i < len(flat_array) - 1:
                f.write(", ")
            if (i + 1) % 12 == 0:
                f.write("\n    ")
        f.write("\n};\n\n")

# ------------------------------------------------------------------------------
# 3. 메인 프로세스 (학습 ➔ 양자화 ➔ 덤프)
# ------------------------------------------------------------------------------
def main():
    # 데이터 로드 (0~1 사이로 정규화)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./MNIST_data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNetLight().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 3-1. 모델 학습 (MNIST는 3~5 에포크면 충분히 수렴)
    print("=== Training Floating-Point Model ===")
    model.train()
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} 완료")

    # 3-2. FP32 정확도 평가
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"FP32 모델 최종 정확도: {correct / 10000 * 100:.2f}%\n")

    # 3-3. 가중치 추출 및 INT8 양자화 적용 (Q_FACTOR = 7, 즉 소수점 7비트 표현)
    print("=== Quantizing Weights ===")
    Q_FACTOR = 7
    SCALE = 2 ** Q_FACTOR

    # 가중치 딕셔너리에서 값 가져오기
    state_dict = model.state_dict()
    conv_w = state_dict['conv.weight'].cpu()
    conv_b = state_dict['conv.bias'].cpu()
    fc_w = state_dict['fc.weight'].cpu()
    fc_b = state_dict['fc.bias'].cpu()

    # 양자화 진행
    q_conv_w = quantize_tensor(conv_w, SCALE).numpy()
    q_conv_b = quantize_tensor(conv_b, SCALE).numpy()
    q_fc_w = quantize_tensor(fc_w, SCALE).numpy()
    # Bias는 보통 곱셈 연산 결과(Scale * Scale)에 맞춰 더 큰 비트(INT16/32)로 잡거나, 
    # 여기서는 구현 단순화를 위해 동일 레벨로 스케일링 후 바이어싱 처리
    q_fc_b = quantize_tensor(fc_b, SCALE).numpy() 

    # 3-4. C언어 헤더 파일로 덤프 생성
    header_path = "./weights/quantized_weights.h"
    # 기존 파일 초기화 및 내보내기
    with open(header_path, 'w') as f:
        f.write("#ifndef QUANTIZED_WEIGHTS_H\n#define QUANTIZED_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"// Q_FACTOR = {Q_FACTOR} (Scale = {SCALE})\n\n")

    dump_to_c_header(header_path, "conv_weights", q_conv_w)
    dump_to_c_header(header_path, "conv_bias", q_conv_b)
    dump_to_c_header(header_path, "fc_weights", q_fc_w)
    dump_to_c_header(header_path, "fc_bias", q_fc_b)

    with open(header_path, 'a') as f:
        f.write("#endif // QUANTIZED_WEIGHTS_H\n")

    print(f"성공: C언어 헤더 파일이 {header_path} 에 저장되었습니다.")

if __name__ == '__main__':
    main()