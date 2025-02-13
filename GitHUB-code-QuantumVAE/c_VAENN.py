import torch
import torch.nn as nn 
import torch.nn.functional as F

class EncoderNN(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=16):
        super(EncoderNN, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim=256
        self.fc_hidden1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_hidden2 = nn.Linear(self.hidden_dim, 128)
        self.fc_mu = nn.Linear(128, self.latent_dim)
        self.fc_log_var = nn.Linear(128, self.latent_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):
        x = x.view(-1, 1, 16, 16)
        x = self.avgpool(x)
        x = x.reshape(-1, 16 * 16 * 4)
        
        x = torch.relu(self.fc_hidden1(x))
        x = torch.relu(self.fc_hidden2(x))
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var
    
class DecoderNN(nn.Module):
    def __init__(self,
                 kernel_size = 4,
                 init_channels = 8,
                 image_channels = 1,
                 latent_dim = 16):
        super(DecoderNN, self).__init__()

        self.fc2 = nn.Linear(latent_dim, 64)
        self.dec1 = nn.ConvTranspose2d(in_channels=64, 
                                       out_channels=init_channels*8, 
                                       kernel_size=kernel_size,
                                       stride=1, 
                                       padding=0)
        self.dec2 = nn.ConvTranspose2d(in_channels=init_channels*8, 
                                       out_channels=init_channels*4, 
                                       kernel_size=kernel_size,
                                       stride=2, 
                                       padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=init_channels*4, 
                                       out_channels=init_channels*2, 
                                       kernel_size=kernel_size,
                                       stride=2, 
                                       padding=1)
        self.dec4 = nn.ConvTranspose2d(in_channels=init_channels*2, 
                                       out_channels=image_channels, 
                                       kernel_size=kernel_size,
                                       stride=2, 
                                       padding=1)
        
    def forward(self, x):
        x = self.fc2(x)
        x = x.view(-1, 64, 1, 1)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))

        return reconstruction
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.Encoder = EncoderNN()
        self.Decoder = DecoderNN()
        
    def reparameterize(self, 
                       mean, 
                       log_variance):
        
        std = torch.exp(0.5*log_variance) 
        eps = torch.randn_like(std) 
        sample = mean + (eps * std) 

        return sample
    
    def forward(self, x):

        mean, log_variance = self.Encoder(x)
        z = self.reparameterize(mean, log_variance)
        images = self.Decoder(z)
        return images, mean, log_variance