# implementation of MSDNet adapted from https://github.com/kalviny/MSDNet-PyTorch
import torch.nn as nn
import torch
import math
import pdb

class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck,
                 bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=1, padding=1, bias=False))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=2, padding=1, bias=False))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU(True))

        self.net = nn.Sequential(*layer)

    def forward(self, x):

        return self.net(x)


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down',
                                bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal',
                                   bottleneck, bnWidth2)

    def forward(self, x):
        res = [x[1],
               self.conv_down(x[0]),
               self.conv_normal(x[1])]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal',
                                   bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0],
               self.conv_normal(x[0])]

        return torch.cat(res, dim=1)

class MSDNFirstLayer(nn.Module):
    def __init__(
        self, nIn, nOut, dataname, grFactor, nScales,
    ):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        if dataname.startswith('cifar'):
            self.layers.append(ConvBasic(nIn, nOut * grFactor[0],
                                         kernel=3, stride=1, padding=1))
        elif dataname.startswith("imagenet"):
            conv = nn.Sequential(
                    nn.Conv2d(nIn, nOut * grFactor[0], 7, 2, 3),
                    nn.BatchNorm2d(nOut * grFactor[0]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)

        nIn = nOut * grFactor[0]

        for i in range(1, nScales):
            self.layers.append(ConvBasic(nIn, nOut * grFactor[i],
                                         kernel=3, stride=2, padding=1))
            nIn = nOut * grFactor[i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)

        return res

class MSDNLayer(nn.Module):
    def __init__(
        self, nIn, nOut, 
        nScales,
        grFactor,
        bottleneck,
        bnFactor, 
        inScales=None, outScales=None
    ):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else nScales
        self.outScales = outScales if outScales is not None else nScales

        self.nScales = nScales
        self.discard = self.inScales - self.outScales

        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()

        if self.discard > 0:
            nIn1 = nIn * grFactor[self.offset - 1]
            nIn2 = nIn * grFactor[self.offset]
            _nOut = nOut * grFactor[self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, bottleneck,
                                              bnFactor[self.offset - 1],
                                              bnFactor[self.offset]))
        else:
            self.layers.append(ConvNormal(nIn * grFactor[self.offset],
                                          nOut * grFactor[self.offset],
                                          bottleneck,
                                          bnFactor[self.offset]))

        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * grFactor[i - 1]
            nIn2 = nIn * grFactor[i]
            _nOut = nOut * grFactor[i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, bottleneck,
                                              bnFactor[i - 1],
                                              bnFactor[i]))

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """
    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res


class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m
        self.linear = nn.Linear(channel, num_classes)

    def forward(self, x, return_features=False):
        res = self.m(x[-1])
        res = res.view(res.size(0), -1)
        if return_features:
            return self.linear(res), res
        else:
            return self.linear(res)

class MSDNet(nn.Module):
    def __init__(
        self, 
        reduction=0.5,
        nBlocks=1,
        nChannels=32,
        base=4,
        stepmode="even", # or lin_grow
        step=1,
        growthRate=6,
        grFactor=[1,2,4],
        prune="max", # or min
        bnFactor=[1,2,4],
        bottleneck=True,
        dataname="imagenet",
        num_classes=1000,
        resolution=224,
        exit_num=None
    ):
        super(MSDNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.steps = [base]
        self.reduction = reduction
        self.nBlocks = nBlocks
        self.nChannels = nChannels
        self.base = base
        self.stepmode = stepmode
        self.step = step
        self.growthRate = growthRate
        self.grFactor = grFactor
        self.prune = prune
        self.bnFactor = bnFactor
        self.bottleneck = bottleneck
        self.dataname = dataname
        self.nScales = len(self.grFactor)
        self.head_locs = {f"head{i+1}": None for i in range(self.nBlocks)}
        self.resolution = resolution
        # this is just for counting MACs and not for actual use
        self.exit_num = exit_num if exit_num is not None else self.nBlocks
        
        n_layers_all, n_layer_curr = self.base, 0
        for i in range(1, self.nBlocks):
            self.steps.append(self.step if self.stepmode == 'even'
                             else self.step * i + 1)
            n_layers_all += self.steps[-1]

        print("building network of steps: ")
        print(self.steps, n_layers_all)

        nIn = self.nChannels
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, self.steps[i],
                                  n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]

            if self.dataname.startswith('cifar100'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * self.grFactor[-1], 100))
            elif self.dataname.startswith('cifar10'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * self.grFactor[-1], 10))
            elif self.dataname == 'imagenet':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * self.grFactor[-1], 1000))
            elif self.dataname == 'imagenet200':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * self.grFactor[-1], 200))
            else:
                self._build_classifier_imagenet(nIn * self.grFactor[-1], num_classes)

        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

        for m in self.classifier:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(
        self, nIn, step, n_layer_all, n_layer_curr
    ):

        layers = [
            MSDNFirstLayer(
                3, nIn, self.dataname,self.grFactor, self.nScales
            )
        ] \
            if n_layer_curr == 0 else []
        for i in range(step):
            n_layer_curr += 1
            inScales = self.nScales
            outScales = self.nScales
            if self.prune == 'min':
                inScales = min(self.nScales, n_layer_all - n_layer_curr + 2)
                outScales = min(self.nScales, n_layer_all - n_layer_curr + 1)
            elif self.prune == 'max':
                interval = math.ceil(1.0 * n_layer_all / self.nScales)
                inScales = self.nScales - math.floor(1.0 * (max(0, n_layer_curr - 2)) / interval)
                outScales = self.nScales - math.floor(1.0 * (n_layer_curr - 1) / interval)
            else:
                raise ValueError

            layers.append(
                MSDNLayer(
                    nIn, self.growthRate, 
                    self.nScales,self.grFactor, self.bottleneck, self.bnFactor,
                    inScales, outScales
                )
            )
            print('|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t|'.format(inScales, outScales, nIn, self.growthRate))

            nIn += self.growthRate
            if self.prune == 'max' and inScales > outScales and \
                    self.reduction > 0:
                offset = self.nScales - outScales
                layers.append(
                    self._build_transition(nIn, math.floor(1.0 * self.reduction * nIn),
                                           outScales, offset))
                _t = nIn
                nIn = math.floor(1.0 * self.reduction * nIn)
                print('|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|'.format(_t, math.floor(1.0 * self.reduction * _t)))
            elif self.prune == 'min' and self.reduction > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = self.nScales - outScales
                layers.append(self._build_transition(nIn, math.floor(1.0 * self.reduction * nIn),
                                                     outScales, offset))

                nIn = math.floor(1.0 * self.reduction * nIn)
                print('|\t\tTransition layer inserted! (min)\t|')
            print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset):
        net = []
        for i in range(outScales):
            net.append(ConvBasic(nIn * self.grFactor[offset + i],
                                 nOut * self.grFactor[offset + i],
                                 kernel=1, stride=1, padding=0))
        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2),
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2)
        )
        return ClassifierModule(conv, nIn, num_classes)

    def forward(self, x, return_features=False):
        # output a dictionary where each head has a key
        # dictionary is ordered automatically as of 3.6
        res = {}
        res_features = {}
        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            if not return_features:
                res[f"head{i+1}"] = self.classifier[i](
                    x, return_features=return_features
                )
            else:
                res[f"head{i+1}"], res_features[f"head{i+1}"] = self.classifier[i](
                    x, return_features=return_features
                )
            if i == self.exit_num-1:
                if return_features:
                    return res, res_features
                else:
                    return res
        if return_features:
            return res, res_features
        else:
            return res

