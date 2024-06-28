import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, device=None):

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Model initialised to '{self.device}'")

    def forward(self, x):
        """
        Forward pass logic to be overridden by subclass specific architecture.
        """
        raise NotImplementedError("Forward pass needs to be defined in subclasses.")


# if __name__ == "__main__":
#     print(torch.__version__)
#     print(torch.cuda.is_available())

#     import matplotlib
#     print(matplotlib.__version__)
#     print(matplotlib.get_backend())
