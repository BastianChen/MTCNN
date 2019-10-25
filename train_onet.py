import nets
import trainer
import torch

if __name__ == '__main__':
    net = nets.ONet()
    if torch.cuda.is_available():
        # trainer = train.Trainer(net, r"F:\Photo_example\CelebA\sample\48", "models/onet.pth", True)
        # trainer = trainer.Trainer(net, r"C:\sample\48", "models/onet.pth", True)
        # trainer = train.Trainer(net, r"C:\new_sample\48", "models/onet.pth", True)
        trainer = trainer.Trainer(net, r"C:\sample\48", "models/onet_residualconv.pth", True)
    else:
        # trainer = train.Trainer(net, r"F:\Photo_example\CelebA\sample\48", "models/onet.pth", False)
        trainer = trainer.Trainer(net, r"C:\sample\48", "models/onet.pth", False)
    trainer.train()
