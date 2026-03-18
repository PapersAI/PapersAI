import torch 
from torch import nn, Tensor


class AlexNet(nn.Module):
    """ 
    Re-Producing of 2012 AlexNet. 

    We follow the implementation of PyTorch version, but we add new checks for images with smaller sizes.


    Max Pooling and ReLU do not have trainable parameters.

    References:
        1. https://d2l.ai/chapter_convolutional-modern/alexnet.html
        2. https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
        3. https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    """
    def __init__(self, 
        input_size: int = 224, 
        cin: int = 3, 
        cout: int = 1000,
        dropout: float = 0.5,
        features_only: bool = False,
        ) -> None:
        """  
        Parameters:
            cin  -- input channels
            cout -- output channels
            dropout -- dropout probability
        """
        super().__init__()

        if input_size > 64:
            # ImageNet options
            k1, s1, p1 = 11, 4, 2  # [224, 224] -> [55, 55]
            k2, p2 = 5, 2
            pool_k, pool_s = 3, 2
        else:
            # smaller input sizes (CIFAR10)
            k1, s1, p1 = 3, 1, 1 
            k2, p2 = 3, 1
            pool_k, pool_s = 2, 2
        
        self.features_only = features_only

        self.features = nn.Sequential(
            # first conv: floor(((h + 2 * padding - 1 * (kernel_size - 1) - 1))/stride + 1)
            # 224 (11, 4, 2): [bs, cin, 224, 224] -> [bs, 64, 55, 55] 
            # 32 (3, 1, 1): [bs, cin, 32, 32] -> [bs, 64, 32, 32]
            nn.Conv2d(cin, 64, kernel_size=k1, stride=s1, padding=p1),
            nn.ReLU(inplace=True),

            # first pool: floor(((h + 2 * padding - 1 * (kernel_size - 1) - 1))/stride + 1)
            # 224 (3, 2): [bs, 64, 55, 55] -> [bs, 64, 27, 27]
            # 32 (2, 2): [bs, 64, 32, 32] -> [bs, 64, 16, 16] 
            nn.MaxPool2d(kernel_size=pool_k, stride=pool_s),  # no training parameters

            # second conv
            # 224 (5, 1, 2): [bs, 64, 27, 27] -> [bs, 192, 27, 27]
            # 32 (3, 1, 1): [bs, 64, 16, 16] -> [bs, 192, 16, 16] 
            nn.Conv2d(64, 192, kernel_size=k2, padding=p2),
            nn.ReLU(inplace=True),

            # second pool
            # 224 (3, 2): [bs, 192, 27, 27] -> [bs, 192, 13, 13]
            # 32 (2, 2): [bs, 192, 16, 16] -> [bs, 192, 8, 8]
            nn.MaxPool2d(kernel_size=pool_k, stride=pool_s),  # no training parameters

            # third conv
            # 224 (3, 1, 1): [bs, 192, 13, 13] -> [bs, 384, 13, 13]
            # 32 (3, 1, 1): [bs, 192, 8, 8] -> [bs, 384, 8, 8]
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  

            # 4th conv
            # 224 (3, 1, 1): [bs, 384, 13, 13] -> [bs, 256, 13, 13]
            # 32 (3, 1, 1): [bs, 384, 8, 8] -> [bs, 256, 8, 8]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 5th conv
            # 224 (3, 1, 1): [bs, 256, 13, 13] -> [bs, 256, 13, 13]
            # 32 (3, 1, 1): [bs, 256, 8, 8] -> [bs, 256, 8, 8]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # third pool
            # 224 (3, 2): [bs, 256, 13, 13] -> [bs, 256, 6, 6]
            # 32 (2, 2): [bs, 256, 8, 8] -> [bs, 256, 4, 4]
            nn.MaxPool2d(kernel_size=pool_k, stride=pool_s),  # no training parameters
        )

        # avg pool
        # [bs, 256, x > 0, x > 0] -> [bs, 256, 6, 6]
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Flatten(),  # [bs, 256 * 6 * 6, 1, 1] -> [bs, 256 * 6 * 6]  dimension changes!
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),  # [bs, 256 * 6 * 6] -> [bs, 4096]
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
            nn.Linear(4096, 4096),  # [bs, 4096] -> [bs, 4096]
            nn.ReLU(inplace=True),

            nn.Linear(4096, cout),  # [bs, 4096] -> [bs, cout]
        )
    
    def forward(self, x: Tensor) -> Tensor:        
        x = self.features(x)
        if not self.features_only:
            x = self.avgpool(x)
            x = self.classifier(x)
        
        return x
    

if __name__ == "__main__":
    net = AlexNet(input_size=32, cin=1)
    x = torch.randn(2, 1, 32, 32)
    y = net(x)
