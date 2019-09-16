import nets
import train
import torch

if __name__ == '__main__':
    net = nets.PNet()
    if torch.cuda.is_available():
        # trainer = train.Trainer(net, r"F:\Photo_example\CelebA\sample\12", "models/pnet.pth", True)
        trainer = train.Trainer(net, r"C:\sample\12", "models/pnet.pth", True)
        # trainer = train.Trainer(net, r"C:\new_sample\12", "models/pnet.pth", True)
    else:
        # trainer = train.Trainer(net, r"F:\Photo_example\CelebA\sample\12", "models/pnet.pth", False)
        trainer = train.Trainer(net, r"C:\sample\12", "models/pnet.pth", False)
    trainer.train()
