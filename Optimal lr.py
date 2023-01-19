import math
def find_lr(model, dataloaders, loss_fn, optimizer, init_value=1e-8, final_value=10.0, use_gpu=True):
    """Слегка модифицированная функция для поиска оптимального learning rate
    функция взята из замечатлеьной книги книги "Ian Pointer - Programming PyTorch
    for Deep Learning - Creating and Deploying Deep Learning Applications-
    O’Reilly Media (2019)
    """
    model.train()
    number_in_epoch = len(dataloaders['train']) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for inputs, labels in dataloaders['train']:
        if use_gpu:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

        batch_num += 1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Crash out if loss explodes

        if batch_num > 1 and loss > 4 * best_loss:
            return log_lrs[10:-5], losses[10:-5]

        # Record the best loss

        if loss < best_loss or batch_num == 1:
            best_loss = loss

        # Store the values

        losses.append(loss)
        log_lrs.append(math.log10(lr))

        # Do the backward pass and optimize

        loss.backward()
        optimizer.step()

        # Update the lr for the next step and store

        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
    return log_lrs[10:-5], losses[10:-5]


# Создаём сеть
model_vgg16_bn = models.vgg16_bn(pretrained=True)

for param in model_vgg16_bn.parameters():
    param.requires_grad = False

for param in model_vgg16_bn.classifier.parameters():
    param.requires_grad = True

# В качестве cost function используем кросс-энтропию
loss_fn = nn.CrossEntropyLoss()

# В качестве оптимизатора AdaShift из репозитория МФТИ
optimizer = torch.optim.AdamW(model_vgg16_bn.classifier.parameters(), lr=1e-3, amsgrad=True)

# Использовать ли GPU
model_vgg16_bn = model_vgg16_bn.cuda()

# подбор оптимального lr для классификатора model_vgg16_bn.classifier
logs, losses = find_lr(model_vgg16_bn, dataloaders, loss_fn, optimizer, init_value=1e-8, final_value=10.0)

# построим график для оптимального подбора lr
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(logs,losses)
ax.set_xlabel("$10^x$")
ax.set_ylabel("loss")