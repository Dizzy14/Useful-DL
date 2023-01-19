# В качестве cost function используем кросс-энтропию
loss_fn = nn.CrossEntropyLoss()

found_lr = lr=1e-3

# Дифференциальное обучение (по группам слоев)у каждой группы свой lr

optimizer = torch.optim.AdamW([
{ 'params': model_vgg16_bn.classifier.parameters(), 'lr': found_lr},
{ 'params': param512, 'lr': found_lr / 3},
{ 'params': param256, 'lr': found_lr / 10},
{ 'params': param128, 'lr': found_lr / 50},
{ 'params': param64, 'lr': found_lr / 100},
], lr=found_lr / 100, amsgrad=True)

# Использовать ли GPU
model_vgg16_bn = model_vgg16_bn.cuda()
