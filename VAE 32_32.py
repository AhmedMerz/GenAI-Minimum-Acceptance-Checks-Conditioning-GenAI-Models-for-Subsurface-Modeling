import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np



# Load your dataset
data_path = 'C:/Users/am224745/Desktop/Ergodic/Training/my_data120_32_32.npz'  # Update this path
data = np.load(data_path)
images = data['my_array1'].astype(np.float32)
mean = images.reshape(1000,32*32).mean(axis=1).reshape(1000,1,1)
std = images.reshape(1000,32*32).std(axis=1).reshape(1000,1,1)
images = (images - mean) / std*0.025 + 0.1
max_value = images.max()  # Find the maximum value to normalize the dataset
min_value = images.min()
images =(images-min_value)/(max_value-min_value)  # Normalize images
images = np.expand_dims(images, axis=1)  # Add channel dimension
train_images, val_images = train_test_split(images, test_size=0.15, random_state=42)

# Convert to PyTorch tensors and create dataloaders
train_tensor = torch.tensor(train_images)
val_tensor = torch.tensor(val_images)
batch_size = 500
train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate_images(model, device, num_images=100):
    model.eval()
    generated_images = []
    with torch.no_grad():
        for _ in range(num_images):
            z = torch.randn(1, 100).to(device)  # Generate random latent vectors
            generated_image = model.decoder(z)  # Generate images from latent vectors
            generated_images.append(generated_image.cpu().numpy())
    model.train()
    return np.concatenate(generated_images, axis=0)
class CNNEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)  # Output: 16x16
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # Output: 8x8
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # Output: 4x4
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # Output: 2x2
        self.bn4 = nn.BatchNorm2d(512)
        self.fc_mu = nn.Linear(512 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(512 * 2 * 2, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(CNNDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        # Adjusted to match the output size of 32x32 from the latent space
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # 8x8
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # 16x16
        self.conv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1) # 32x32

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 128, 4, 4)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = torch.sigmoid(self.conv3(z))
        return z

class VAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(VAE, self).__init__()
        self.encoder = CNNEncoder(latent_dim)
        self.decoder = CNNDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # Change from BCE to MSE
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence remains the same
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE +  KLD

model = VAE(latent_dim=100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def validate(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, in val_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            val_loss += loss_function(recon_batch, data, mu, logvar).item()
    val_loss /= len(val_loader.dataset)
    print(f'====> Val set loss: {val_loss:.4f}')
OA2= []
num_epochs = 10000
for epoch in range(1, num_epochs + 1):
    train(epoch)  # Train your model
    validate(epoch)  # Validate your model
    
    # Every 100 epochs, generate images and calculate the overlap coefficient
    if epoch % 100 == 0:
        generated_images = generate_images(model, device, num_images=100)
        generated_images = generated_images.flatten()*(max_value-min_value) + min_value   # Assuming the same normalization as training data

        # Assuming 'images' contains the training images flattened and normalized
        data1 = images.flatten()*(max_value-min_value) + min_value # Use the training set for comparison
        hist_range = (0, 1)  # Adjust this range based on your data's values

        # Create normalized histograms for both distributions
        hist1, bins = np.histogram(data1, bins=35, range= hist_range, density=True)
        hist2, _ = np.histogram(generated_images, bins=bins, range= hist_range, density=True)
        
        bin_width = np.diff(bins)

        # Calculate the overlapping area
        overlap = np.sum(np.minimum(hist1, hist2) * bin_width)
        total_area_new_data = np.sum(hist1 * bin_width)
        overlap_percentage = (overlap / total_area_new_data) * 100
        OA2.append(overlap_percentage)
        
        print(f'Epoch: {epoch}, Overlap Percentage: {overlap_percentage:.2f}%')

    