import nets
import trainer
import torch

if __name__ == '__main__':
    net = nets.RNet()
    if torch.cuda.is_available():
        # trainer = train.Trainer(net, r"F:\Photo_example\CelebA\sample\24", "models/rnet.pth", True)
        # trainer = trainer.Trainer(net, r"C:\sample\24", "models/rnet.pth", True)
        # trainer = train.Trainer(net, r"C:\new_sample\24", "models/rnet.pth", True)
        trainer = trainer.Trainer(net, r"C:\sample\24", "models/rnet_depthwiseconv.pth", True)
    else:
        # trainer = train.Trainer(net, r"F:\Photo_example\CelebA\sample\24", "models/rnet.pth", False)
        trainer = trainer.Trainer(net, r"C:\sample\24", "models/rnet.pth", False)
    trainer.train()
