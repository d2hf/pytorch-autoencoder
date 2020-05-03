class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3,stride= 1, padding= 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
            nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
            nn.ReLU(True)
            )

        self.decoder = nn.Sequential(
            nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
            nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            
            nn.Conv2d(16, 16, 3,stride= 1, padding= 1),
            nn.ReLU(True),
            
            nn.Conv2d(16, 1, 1,stride= 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        output = self.decoder(x)
        return output
