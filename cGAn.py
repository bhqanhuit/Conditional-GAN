import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision.utils import save_image


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_path_data = "load_data/fashion_mnist/fashion-mnist_train.csv"
valid_path_data = "load_data/fashion_mnist/fashion-mnist_test.csv"

img_size = 28
batch_size = 64
z_dim = 100

EPOCH = 3
learning_rate = 1e-4

class Fashion_mnist(Dataset):
    def __init__(self, path, img_size, transform=None):
        fashion_df = pd.read_csv(path)
        self.transform = transform
        self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, img_size, img_size)
        self.labels = fashion_df["label"].values
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) :
        label = self.labels[index]
        image = self.images[index]
        image = Image.fromarray(self.images[index])

        if self.transform:
            image = self.transform(image)
        
        return image, label

class_list = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_num = len(class_list)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

dataset = Fashion_mnist(train_path_data, img_size=img_size, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# print(dataset[0][0][0, :, :].shape)
# plt.imshow(dataset[0][0][0, :, :], cmap="gray")
# plt.show()

# for images, labels in dataloader:
#     fig, ax = plt.subplots(figsize=(18,10))
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.imshow(make_grid(images, nrow=16).permute(1,2,0))
#     plt.show()
#     break


class Generator(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.z_dim = 100
        self.img_size = 28
        self.class_num = class_num
        
        self.label_emb = nn.Embedding(class_num, class_num)

        self.model = nn.Sequential(
            nn.Linear(self.z_dim + self.class_num, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.img_size*self.img_size),
            nn.Tanh()
        )

    def forward(self, z, label):
        # label = torch.tensor(label).to(torch.int64)
        z = z.view(-1, self.z_dim)
        label = (self.label_emb(label))

        res = torch.cat([z, label], 1)
        # print(res.shape)
        res = self.model(res)
        return res.view(-1, self.img_size, self.img_size)
    
class Discriminator(nn.Module):
    def __init__(self, img_size, class_num):
        super().__init__()
        
        self.img_size = img_size        
        self.label_emb = nn.Embedding(class_num, class_num)

        self.model = nn.Sequential(
            nn.Linear(self.img_size*self.img_size + class_num, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        # label = torch.tensor(label).to(torch.int64)
        x = x.view(-1, self.img_size*self.img_size)
        label = (self.label_emb(label))

        res = torch.cat([x, label], 1)
        res = self.model(res)
        return res.squeeze()
    


if (__name__ =="__main__"):
    gen = Generator(class_num=class_num)
    dis = Discriminator(img_size=28, class_num=class_num)

    gen.to(device)
    dis.to(device)
    # summary(gen, [(1, 100), (1, 1)])
    # summary(dis, [(1, 28*28), (1, 1)])

    criterion = nn.BCELoss()
    criterion.to(device)
    gen_optim = torch.optim.Adam(gen.parameters(), lr=learning_rate)
    dis_optim = torch.optim.Adam(dis.parameters(), lr=learning_rate)
    training_df = {"epoch":[], "Batch":[], "D_loss":[], "G_loss":[]}


    for epoch in range(EPOCH):  
        for i, (real_img, real_label) in enumerate(dataloader):
            # gen.train()

            # dis_optim.zero_grad()
            # D_real = dis(real_img, real_label)
            # D_loss_real = criterion(D_real, Variable(torch.ones(batch_size)).to(device))
            
            # z = Variable(torch.randn(batch_size, 100)).to(device)
            # fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_num, batch_size))).to(device)
            # fake_imgs = gen(z, fake_labels)
            # D_fake = dis(fake_imgs, fake_labels)
            # D_loss_fake = criterion(D_fake, Variable(torch.zeros(batch_size)).to(device))
            # D_loss = D_loss_real + D_loss_fake
            # D_loss.backward()
            # dis_optim.step()


            # gen_optim.zero_grad()
            # z = Variable(torch.randn(batch_size, 100)).to(device)
            # fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_num, batch_size))).to(device)
            # fake_imgs = gen(z, fake_labels)
            # G_validity = dis(fake_imgs, fake_labels)
            # G_loss = criterion(G_validity, Variable(torch.ones(batch_size)).to(device))
            # G_loss.backward()
            # gen_optim.step()






            gen.train()
            dis.zero_grad()
            z_noise = torch.randn((batch_size, z_dim)).to(device)
            fake_label = torch.randint(size=(batch_size, 1), low=0, high=9).squeeze().to(device)    
    
            real = real_img
            fake = gen(z_noise, fake_label)

            D_real = dis(real, real_label)
            D_loss_real = criterion(D_real, torch.ones_like(D_real)).to(device)
            D_fake = dis(fake, fake_label)  
            D_loss_fake = criterion(D_fake, torch.zeros_like(D_fake)).to(device)
            D_loss = (D_loss_real + D_loss_fake)

            D_loss.backward()
            dis_optim.step()
            

            z_noise = torch.randn((batch_size, z_dim)).to(device)
            fake_label = torch.randint(size=(batch_size, 1), low=0, high=9).squeeze().to(device)   
            fake = gen(z_noise, fake_label)

            gen_optim.zero_grad()
            gen_out = dis(fake, fake_label)
            gen_loss = criterion(gen_out, torch.ones_like(gen_out)).to(device)
            gen_loss.backward()
            gen_optim.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, EPOCH, i, len(dataloader), D_loss.item(), gen_loss.item())
            )

            training_df["epoch"].append((epoch, EPOCH))
            training_df["Batch"].append((i, len(dataloader)))
            training_df["D_loss"].append(D_loss.item())
            training_df["G_loss"].append(gen_loss.item())
            # if (i > 10): break

        
        z_sample = torch.randn((6, 100)).to(device)
        sample_label = torch.randint(size=(6, 1), low=0, high=9).squeeze().to(device)

        sample_img = gen(z_sample, sample_label).unsqueeze(1)
        # grid = make_grid(sample_img, nrow=3, normalize=True).permute(1,2,0)
        # print(sample_img.permute(0,2,3,1).shape)
        save_image(sample_img, "images/%d.png" % epoch, nrow=3, normalize=True)

        torch.save({
                'epoch': epoch,
                'model_state_dict': gen.state_dict(),
                'optimizer_state_dict': gen_optim.state_dict(),
                'loss': gen_loss,
                }, "save_model/cgan.pt")

    training_df = pd.DataFrame(training_df)
    training_df.to_csv("training_log/cgan.csv")
    


