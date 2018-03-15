import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from PIL import Image
import os
import net
import utils


# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        G.div_(h*w)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)


# Stylize
class NeuralStyle(object):
    def __init__(self, option):
        # Parameters
        self.image_dir = option.image_dir
        self.model_dir = option.model_dir
        self.result_dir = option.result_dir
        self.content_img = option.content_img
        self.style_img = option.style_img
        self.lr_img_size = option.lr_img_size
        self.max_iter = option.max_iter
        self.show_iter = option.show_iter

        # load images, ordered as [style_image, content_image]
        img_dirs = [self.image_dir, self.image_dir]
        img_names = [self.style_img, self.content_img]
        self.imgs = [Image.open(img_dirs[i] + name) for i, name in enumerate(img_names)]

        # get network
        self.vgg = net.VGG()
        self.vgg.load_state_dict(torch.load(self.model_dir + 'vgg_conv.pth'))
        for param in self.vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg.cuda()

        # define layers, loss functions, weights and compute optimization targets
        self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.content_layers = ['r42']
        self.loss_layers = self.style_layers + self.content_layers
        self.loss_fns = [GramMSELoss()] * len(self.style_layers) + [nn.MSELoss()] * len(self.content_layers)
        if torch.cuda.is_available():
            self.loss_fns = [loss_fn.cuda() for loss_fn in self.loss_fns]

        # these are good weights settings:
        style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
        content_weights = [1e0]
        self.weights = style_weights + content_weights

    def stylize(self):
        min_img_size = min(min(self.imgs[0].size), min(self.imgs[1].size))
        hr_img_size = min(self.imgs[1].size)

        imgs_torch = [utils.get_prep(self.lr_img_size)(img) for img in self.imgs]
        if torch.cuda.is_available():
            imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
        else:
            imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]

        style_image, content_image = imgs_torch

        # opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
        opt_img = Variable(content_image.data.clone(), requires_grad=True)

        # compute optimization targets
        style_targets = [GramMatrix()(A).detach() for A in self.vgg(style_image, self.style_layers)]
        content_targets = [A.detach() for A in self.vgg(content_image, self.content_layers)]
        targets = style_targets + content_targets

        # run style transfer
        optimizer = optim.LBFGS([opt_img])
        n_iter = [0]

        while n_iter[0] <= self.max_iter:

            def closure():
                optimizer.zero_grad()
                out = self.vgg(opt_img, self.loss_layers)
                layer_losses = [self.weights[a] * self.loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
                loss = sum(layer_losses)
                loss.backward()
                n_iter[0] += 1
                # print loss
                if n_iter[0] % self.show_iter == (self.show_iter - 1):
                    print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.data[0]))
                # print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
                return loss

            optimizer.step(closure)

        out_img = utils.postp(opt_img.data[0].cpu().squeeze())

        # make the image high-resolution as described in
        # "Controlling Perceptual Factors in Neural Style Transfer", Gatys et al.
        # (https://arxiv.org/abs/1611.07865)
        if min_img_size > self.lr_img_size:
            # prep hr images
            imgs_torch = [utils.get_prep(hr_img_size)(img) for img in self.imgs]
            if torch.cuda.is_available():
                imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
            else:
                imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
            style_image, content_image = imgs_torch

            # now initialise with upsampled lowres result
            opt_img = utils.get_prep(hr_img_size)(out_img).unsqueeze(0)
            opt_img = Variable(opt_img.type_as(content_image.data), requires_grad=True)

            # compute hr targets
            style_targets = [GramMatrix()(A).detach() for A in self.vgg(style_image, self.style_layers)]
            content_targets = [A.detach() for A in self.vgg(content_image, self.content_layers)]
            targets = style_targets + content_targets

            # run style transfer for high res
            optimizer = optim.LBFGS([opt_img])
            n_iter = [0]
            while n_iter[0] <= (self.max_iter // 2.5):

                def closure():
                    optimizer.zero_grad()
                    out = self.vgg(opt_img, self.loss_layers)
                    layer_losses = [self.weights[a] * self.loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
                    loss = sum(layer_losses)
                    loss.backward()
                    n_iter[0] += 1
                    # print loss
                    if n_iter[0] % self.show_iter == (self.show_iter - 1):
                        print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.data[0]))
                    # print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
                    return loss

                optimizer.step(closure)

        print('Stylization finished.')

        # save result
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        out_img = utils.postp(opt_img.data[0].cpu().squeeze())
        out_img.save(self.result_dir + 'output.png')
