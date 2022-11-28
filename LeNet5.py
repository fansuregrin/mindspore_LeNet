from mindspore import nn, ops


class LeNet5(nn.Cell):
    def __init__(self, n_classes, in_channels):
        super().__init__()
        self.feature_extractor = nn.SequentialCell(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, pad_mode='valid'), # 28x28xin_ch -> 24x24x6
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), # 24x24x6 -> 12x12x6
            nn.Conv2d(6, 16, kernel_size=5, stride=1, pad_mode='valid'), # 12x12x6 -> 8x18x16
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), # 8x8x16 -> 4x4x16
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell(
            nn.Dense(4*4*16, 120),
            nn.ReLU(),
            nn.Dense(120, 84),
            nn.ReLU(),
            nn.Dense(84, n_classes)
        )

    def construct(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        # probs = nn.Softmax(logits)

        return logits#, probs


def train(model, dataset, loss_fn, optimizer, logging=None):
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # Define forward function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            log_string = f"loss: {loss:>7f} [{current:>3d}/{size:>3d}]"
            print(log_string)
            if logging is not None:
                logging.info(log_string)

def test(model, dataset, loss_fn, logging=None):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    log_string = f"Test: \nAccuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f}\n"
    print(log_string)
    if logging is not None:
        logging.info(log_string)

