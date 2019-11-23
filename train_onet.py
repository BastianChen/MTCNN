import nets
import trainer
import torch

if __name__ == '__main__':
    net = nets.ONet()
    if torch.cuda.is_available():
        # trainer = trainer.Trainer(net, r"C:\sample\48", "models/onet_residualconv.pth", True)
        trainer = trainer.Trainer(net, r"C:\sample\celeba\48", "models/onet_attention.pth", True)
    else:
        trainer = trainer.Trainer(net, r"C:\sample\celeba\48", "models/onet.pth", False)
    trainer.train()
