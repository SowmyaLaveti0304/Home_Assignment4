import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

def main():
    #–– Device & Hyperparams ––#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    batch_size = 256
    lr = 2e-4
    epochs = 50
    os.makedirs("samples_opt", exist_ok=True)

    #–– DataLoader ––#
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    mnist = datasets.MNIST("data", train=True, download=True, transform=transform)
    # Use fewer workers (or 0) on macOS / CPU
    num_workers = 4 if device.type=="cuda" else 0
    pin_memory = True if device.type=="cuda" else False

    loader = DataLoader(
        mnist, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    #–– Models ––#
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.ReLU(True),
                nn.Linear(256, 512),         nn.BatchNorm1d(512), nn.ReLU(True),
                nn.Linear(512, 1024),        nn.BatchNorm1d(1024), nn.ReLU(True),
                nn.Linear(1024, 28*28),      nn.Tanh()
            )
        def forward(self, z):
            return self.net(z).view(-1,1,28,28)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                nn.Linear(1024, 512),  nn.LeakyReLU(0.2), nn.Dropout(0.3),
                nn.Linear(512, 256),   nn.LeakyReLU(0.2), nn.Dropout(0.3),
                nn.Linear(256, 1),     nn.Sigmoid()
            )
        def forward(self, img):
            return self.net(img)

    G = Generator().to(device)
    D = Discriminator().to(device)

    #–– Optimizers & AMP ––#
    criterion = nn.BCELoss()
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    scalerD = GradScaler(enabled=(device.type=="cuda"))
    scalerG = GradScaler(enabled=(device.type=="cuda"))

    fixed_noise = torch.randn(64, latent_dim, device=device)
    g_losses, d_losses = [], []

    for epoch in range(1, epochs+1):
        for real_imgs, _ in loader:
            real_imgs = real_imgs.to(device, non_blocking=True)
            bs = real_imgs.size(0)

            # — Train Discriminator —#
            noise = torch.randn(bs, latent_dim, device=device)
            fake_imgs = G(noise)

            real_labels = torch.ones(bs,1, device=device)
            fake_labels = torch.zeros(bs,1, device=device)

            D.zero_grad()
            with autocast(enabled=(device.type=="cuda")):
                loss_real = criterion(D(real_imgs), real_labels)
                loss_fake = criterion(D(fake_imgs.detach()), fake_labels)
                lossD = loss_real + loss_fake
            scalerD.scale(lossD).backward()
            scalerD.step(optD)
            scalerD.update()

            # — Train Generator —#
            G.zero_grad()
            with autocast(enabled=(device.type=="cuda")):
                lossG = criterion(D(fake_imgs), real_labels)
            scalerG.scale(lossG).backward()
            scalerG.step(optG)
            scalerG.update()

            g_losses.append(lossG.item())
            d_losses.append(lossD.item())

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                samples = G(fixed_noise).cpu()
                utils.save_image(
                    samples,
                    f"samples_opt/epoch_{epoch}.png",
                    nrow=8, normalize=True, value_range=(-1,1)
                )
            print(f"[Epoch {epoch}] Saved samples.")

    #–– Plot Losses ––#
    plt.figure(figsize=(8,5))
    plt.plot(d_losses, label="D")
    plt.plot(g_losses, label="G")
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.legend(); plt.title("Loss curves")
    plt.tight_layout()
    plt.savefig("loss_curve_opt.png")
    plt.show()

if __name__ == "__main__":
    main()
