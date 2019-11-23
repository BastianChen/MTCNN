import nets
import trainer
import torch

if __name__ == '__main__':
    net = nets.PNet()
    if torch.cuda.is_available():
        # trainer = train.Trainer(net, r"F:\Photo_example\CelebA\sample\12", "models/pnet.pth", True)
        # trainer = trainer.Trainer(net, r"C:\sample\12", "models/pnet_depthwiseconv.pth", True)
        # trainer = trainer.Trainer(net, r"C:\sample\celeba\12", "models/pnet_normal_data_enhancement.pth", True)
        trainer = trainer.Trainer(net, r"C:\sample\celeba\12", "models/pnet_attention.pth", True)
    else:
        trainer = trainer.Trainer(net, r"C:\sample\celeba\12", "models/pnet.pth", False)
    trainer.train()
