from cGAn import Generator
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

model = Generator(class_num=10)
optimizer = torch.optim.Adam(model.parameters())

checkpoint = torch.load("save_model/cgan.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

z_sample = torch.randn((6, 100))
sample_label = torch.randint(size=(6, 1), low=0, high=9).squeeze()

sample_img = model(z_sample, sample_label).unsqueeze(1)
grid = make_grid(sample_img, nrow=3, normalize=True).permute(1,2,0)

plt.imshow(grid)
plt.show()