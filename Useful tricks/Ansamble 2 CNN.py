class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(classes_number * 2, classes_number)

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x


# Зададим путь для загрузки моделей!
path_vgg16_bn = '/путь_до_весов_модели/vgg16_bn.pth'
path_resnet50 = '/путь_до_весов_модели/resnet50.pth'

# Загружаем state dicts
model_vgg16_bn.load_state_dict(torch.load(path_vgg16_bn))
model_resnet50.load_state_dict(torch.load(path_resnet50))


model_ensemble = MyEnsemble(model_vgg16_bn, model_resnet50)

# замораживаем параметры (веса) не входящие в layers_to_unfreeze
for param in model_ensemble.parameters():
    param.requires_grad = False

for param in model_ensemble.classifier.parameters():
    param.requires_grad = True


# посмотрим какие параметры учим у model_ensemble
# print_learn_params(model_ensemble)

Params to learn:
	 classifier.weight
	 classifier.bias


