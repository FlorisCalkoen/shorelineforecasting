import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Simple RNN (LSTM) network.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


def train_model(model, dataloaders, model_configs):
    """
    Training loop for NN network.
    :param model:
    :param dataloaders:
    :param model_configs:
    :return:
    """
    # loss, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model_configs['learning_rate'])
    criterion = nn.MSELoss()
    model = model.double()

    # train
    for epoch in range(model_configs['epochs']):
        running_loss_train = 0
        running_loss_test = 0

        for i, (x1, y1) in enumerate(dataloaders['train']):
            # input size: (batch, seq_len, input_size)
            t_x1 = x1.view(model_configs['batch_size'], model_configs['train_window'], model_configs['input_size'])
            # ouput shape: (batch, seq_len, input_size)
            output = model(t_x1.double())
            loss_train = criterion(output, y1)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            running_loss_train += loss_train.item()

        for i, (x2, y2) in enumerate(dataloaders['val']):
            t_x2 = x2.view(model_configs['batch_size'], model_configs['train_window'], model_configs['input_size'])
            prediction = model(t_x2.double())
            loss_test = criterion(prediction, y2)
            running_loss_test += loss_test.item()
        print('Epoch {} Train Loss:{}, Val Loss:{}'.format(epoch + 1, running_loss_train, running_loss_test))
    print('Finish training')


def inference_model(model, dataloader, model_configs):
    """

    :param model:
    :param dataloaders:
    :param model_configs:
    :return:
    """
    model = model.double()

    model.eval()
    with torch.no_grad():
        # empty torch tensor to store result (row, total_seq_length, input_size)
        result = torch.empty(0, model_configs['forecast_size'], model_configs['input_size'])
        result = result.double()
        for i, (x, y) in enumerate(dataloader):
            # sequence input (x) to (batch_size, seq_length, input size)
            t_x = x.view(model_configs['batch_size'], model_configs['train_window'], model_configs['input_size'])
            for i in range(model_configs['horizon']):
                forecast_input = t_x[:, -model_configs['train_window']:, :]
                # (batch_size, seq_length)
                out = model(forecast_input.double())
                # (batch_size, seq length, input_size)
                out = out.unsqueeze(2)
                # (batch_size, seq_length, input_size)
                t_x = torch.cat((t_x, out), dim=1)

            # store in dataframe
            result = torch.cat((result, t_x))
        return result
