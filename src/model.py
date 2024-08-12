import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture

def depth_wise_conv(in_channels, out_channels, stride):
    
    return nn.Sequential(
        
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(inplace=True),
      
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
    )
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            # Depthwise separable convolutions
            depth_wise_conv(32, 64, 1),
            depth_wise_conv(64, 128, 2),
            depth_wise_conv(128, 128, 1),
            depth_wise_conv(128, 256, 2),
            depth_wise_conv(256, 256, 1),
            depth_wise_conv(256, 512, 2),
            
            # Intermediate depthwise layers
            depth_wise_conv(512, 512, 1),
            depth_wise_conv(512, 512, 1),
            depth_wise_conv(512, 512, 1),
            depth_wise_conv(512, 512, 1),
            depth_wise_conv(512, 512, 1),
            
            # Last layers
            depth_wise_conv(512, 1024, 2),
            depth_wise_conv(1024, 1024, 1),
            
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
  
        x = self.model(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        
        return x
    


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
