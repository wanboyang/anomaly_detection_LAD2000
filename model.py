import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.autograd import Variable
import torch.nn.functional as f

# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        return hidden, cell


class ano_classification_head(nn.Module):
    def __init__(self, ano_classes):
        super(ano_classification_head, self).__init__()
        self.fc_1 = nn.Linear(1024, 512, bias=False)
        self.fc_2 = nn.Linear(512, ano_classes, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.fc_2(x)
        return x


class ano_regression_head(nn.Module):
    def __init__(self, clip_number, clip_frames=16):
        super(ano_regression_head, self).__init__()
        self.clip_number = clip_number
        self.clip_frames = clip_frames
        self.fc_1 = nn.Linear(1024, 512, bias=False)
        self.fc_2 = nn.Linear(512, clip_number * clip_frames, bias=False)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.sigmoid(self.fc_2(x))
        return x




class AED(nn.Module):
    def __init__(self, clip_number, input_size, hidden_size, ano_classes=14):
        super(AED, self).__init__()
        self.clip_number = clip_number
        self.hidden_size = hidden_size
        self.convlstm1 = ConvLSTMCell(input_size, hidden_size)
        # self.convlstm2 = ConvLSTMCell(hidden_size, hidden_size)
        self.fc_down = nn.Linear(2048, 1024)
        self.act = nn.ReLU()
        self.cls_head = ano_classification_head(ano_classes=ano_classes)
        self.reg_head = ano_regression_head(clip_number=clip_number)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.shape[0], x.shape[1], 512, 4, 4)
        T = x.shape[0]
        batch_size = x.shape[1]
        spatial_size = x.size()[3:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        prev_state = [
            Variable(x.new_zeros(state_size)),
            Variable(x.new_zeros(state_size))
        ]
        for t in range(0, T):
            prev_state[0], prev_state[1] = self.convlstm1(x[t], prev_state)
        x = prev_state[0].view(batch_size,-1)
        x = self.act(self.fc_down(x))
        cls_scores = self.cls_head(x)
        reg_scores = self.reg_head(x)
        return cls_scores, reg_scores




class AED_T(nn.Module):
    def __init__(self, clip_number, input_size, hidden_size, ano_classes=14):
        super(AED_T, self).__init__()
        self.num_layers = 2
        self.clip_number = clip_number
        self.hidden_size = hidden_size
        self.convlstm1 = ConvLSTMCell(input_size, hidden_size)
        self.convlstm2 = ConvLSTMCell(hidden_size, hidden_size)
        self.fc_down = nn.Linear(2048, 1024)
        self.act = nn.ReLU()
        self.cls_head = ano_classification_head(ano_classes=ano_classes)
        self.reg_head = ano_regression_head(clip_number=clip_number)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.shape[0], x.shape[1], -1, 4, 4)
        T = x.shape[0]
        batch_size = x.shape[1]
        spatial_size = x.size()[3:]
        state_size = [self.num_layers] + [batch_size, self.hidden_size] + list(spatial_size)
        prev_state = [
            Variable(x.new_zeros(state_size)),
            Variable(x.new_zeros(state_size))
        ]
        for t in range(0, T):
            h1_t, c1_t = self.convlstm1(x[t], prev_state[0])
            h2_t, c2_t = self.convlstm2(h1_t, prev_state[1])
            prev_state = [torch.stack([h1_t, c1_t]), torch.stack([h2_t, c2_t])]
        x = prev_state[1][0].view(batch_size, -1)
        x = self.act(self.fc_down(x))
        cls_scores = self.cls_head(x)
        reg_scores = self.reg_head(x)
        return cls_scores, reg_scores



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


def model_generater(model_name, feature_size, arg=None):
    if model_name == 'AED':
        model = AED(clip_number=arg.max_seqlen, input_size=feature_size//16, hidden_size=128, ano_classes=arg.ano_class)
    elif model_name == 'AED_T':
        model = AED_T(clip_number=arg.max_seqlen, input_size=feature_size//16, hidden_size=128, ano_classes=arg.ano_class)
    else:
        raise ('model_name is out of option')
    return model