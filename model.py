import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.autograd import Variable
import torch.nn.functional as f

# Define some constants / 定义常量
KERNEL_SIZE = 3  # 卷积核大小 / Convolution kernel size
PADDING = KERNEL_SIZE // 2  # 填充大小 / Padding size

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell / 生成卷积LSTM单元
    用于处理时空特征的循环神经网络 / Recurrent neural network for spatio-temporal feature processing
    """

    def __init__(self, input_size, hidden_size):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size  # 输入特征维度 / Input feature dimension
        self.hidden_size = hidden_size  # 隐藏状态维度 / Hidden state dimension
        # 门控卷积层：输入+隐藏状态 -> 4*隐藏状态 / Gated convolution: input+hidden -> 4*hidden
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

    def forward(self, input_, prev_state):
        """
        Forward pass of ConvLSTM cell / ConvLSTM单元的前向传播
        Args:
            input_: 当前时间步的输入 / Input at current time step
            prev_state: 前一个时间步的状态 (hidden, cell) / Previous state (hidden, cell)
        Returns:
            hidden: 当前隐藏状态 / Current hidden state
            cell: 当前细胞状态 / Current cell state
        """

        prev_hidden, prev_cell = prev_state  # 解包前一个状态 / Unpack previous state

        # 拼接输入和隐藏状态 / Concatenate input and hidden state
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)  # 计算门控值 / Compute gate values

        # 将门控值分割为四个门 / Split gate values into four gates
        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # 应用sigmoid激活函数 / Apply sigmoid activation
        in_gate = f.sigmoid(in_gate)        # 输入门 / Input gate
        remember_gate = f.sigmoid(remember_gate)  # 遗忘门 / Forget gate
        out_gate = f.sigmoid(out_gate)      # 输出门 / Output gate

        # 应用tanh激活函数 / Apply tanh activation
        cell_gate = f.tanh(cell_gate)       # 候选细胞状态 / Candidate cell state

        # 计算当前细胞状态和隐藏状态 / Compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)  # 更新细胞状态 / Update cell state
        hidden = out_gate * f.tanh(cell)    # 计算隐藏状态 / Compute hidden state

        return hidden, cell


class ano_classification_head(nn.Module):
    """
    Classification head for anomaly detection / 异常检测分类头
    Predicts anomaly categories / 预测异常类别
    """
    def __init__(self, ano_classes):
        super(ano_classification_head, self).__init__()
        self.fc_1 = nn.Linear(1024, 512, bias=False)  # 全连接层1 / Fully connected layer 1
        self.fc_2 = nn.Linear(512, ano_classes, bias=False)  # 全连接层2 / Fully connected layer 2
        self.act = nn.ReLU()  # 激活函数 / Activation function

    def forward(self, x):
        """
        Forward pass for classification / 分类前向传播
        Args:
            x: 输入特征 / Input features
        Returns:
            分类得分 / Classification scores
        """
        x = self.act(self.fc_1(x))  # 全连接层1 + ReLU / FC1 + ReLU
        x = self.fc_2(x)  # 全连接层2 / FC2
        return x


class ano_regression_head(nn.Module):
    """
    Regression head for anomaly detection / 异常检测回归头
    Predicts anomaly scores for each frame / 预测每帧的异常得分
    """
    def __init__(self, clip_number, clip_frames=16):
        super(ano_regression_head, self).__init__()
        self.clip_number = clip_number  # 视频片段数量 / Number of video clips
        self.clip_frames = clip_frames  # 每个片段的帧数 / Frames per clip
        self.fc_1 = nn.Linear(1024, 512, bias=False)  # 全连接层1 / Fully connected layer 1
        self.fc_2 = nn.Linear(512, clip_number * clip_frames, bias=False)  # 全连接层2 / FC2
        self.act = nn.ReLU()  # ReLU激活函数 / ReLU activation
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数 / Sigmoid activation

    def forward(self, x):
        """
        Forward pass for regression / 回归前向传播
        Args:
            x: 输入特征 / Input features
        Returns:
            回归得分 (0-1之间) / Regression scores (between 0-1)
        """
        x = self.act(self.fc_1(x))  # 全连接层1 + ReLU / FC1 + ReLU
        x = self.sigmoid(self.fc_2(x))  # 全连接层2 + Sigmoid / FC2 + Sigmoid
        return x




class AED(nn.Module):
    """
    Anomaly Event Detection Model / 异常事件检测模型
    Single-layer ConvLSTM architecture / 单层ConvLSTM架构
    """
    def __init__(self, clip_number, input_size, hidden_size, ano_classes=14):
        super(AED, self).__init__()
        self.clip_number = clip_number  # 视频片段数量 / Number of video clips
        self.hidden_size = hidden_size  # 隐藏状态维度 / Hidden state dimension
        self.convlstm1 = ConvLSTMCell(input_size, hidden_size)  # 单层ConvLSTM / Single-layer ConvLSTM
        # self.convlstm2 = ConvLSTMCell(hidden_size, hidden_size)  # 注释掉的第二层 / Commented second layer
        self.fc_down = nn.Linear(2048, 1024)  # 降维全连接层 / Dimensionality reduction FC
        self.act = nn.ReLU()  # 激活函数 / Activation function
        self.cls_head = ano_classification_head(ano_classes=ano_classes)  # 分类头 / Classification head
        self.reg_head = ano_regression_head(clip_number=clip_number)  # 回归头 / Regression head

    def forward(self, x):
        """
        Forward pass of AED model / AED模型前向传播
        Args:
            x: 输入特征序列 / Input feature sequence
        Returns:
            cls_scores: 分类得分 / Classification scores
            reg_scores: 回归得分 / Regression scores
        """
        # 如果输入是3D张量，重塑为5D / If input is 3D tensor, reshape to 5D
        if len(x.shape) == 3:
            x = x.view(x.shape[0], x.shape[1], 512, 4, 4)
        
        T = x.shape[0]  # 时间步数 / Number of time steps
        batch_size = x.shape[1]  # 批大小 / Batch size
        spatial_size = x.size()[3:]  # 空间尺寸 / Spatial dimensions
        
        # 初始化状态 / Initialize state
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        prev_state = [
            Variable(x.new_zeros(state_size)),  # 隐藏状态 / Hidden state
            Variable(x.new_zeros(state_size))   # 细胞状态 / Cell state
        ]
        
        # 时间序列处理 / Time series processing
        for t in range(0, T):
            prev_state[0], prev_state[1] = self.convlstm1(x[t], prev_state)
        
        # 特征处理和预测 / Feature processing and prediction
        x = prev_state[0].view(batch_size,-1)  # 展平 / Flatten
        x = self.act(self.fc_down(x))  # 降维 + 激活 / Dimension reduction + activation
        cls_scores = self.cls_head(x)  # 分类预测 / Classification prediction
        reg_scores = self.reg_head(x)  # 回归预测 / Regression prediction
        
        return cls_scores, reg_scores




class AED_T(nn.Module):
    """
    Anomaly Event Detection Model with Two Layers / 双层异常事件检测模型
    Two-layer ConvLSTM architecture / 双层ConvLSTM架构
    """
    def __init__(self, clip_number, input_size, hidden_size, ano_classes=14):
        super(AED_T, self).__init__()
        self.num_layers = 2  # 层数 / Number of layers
        self.clip_number = clip_number  # 视频片段数量 / Number of video clips
        self.hidden_size = hidden_size  # 隐藏状态维度 / Hidden state dimension
        self.convlstm1 = ConvLSTMCell(input_size, hidden_size)  # 第一层ConvLSTM / First ConvLSTM layer
        self.convlstm2 = ConvLSTMCell(hidden_size, hidden_size)  # 第二层ConvLSTM / Second ConvLSTM layer
        self.fc_down = nn.Linear(2048, 1024)  # 降维全连接层 / Dimensionality reduction FC
        self.act = nn.ReLU()  # 激活函数 / Activation function
        self.cls_head = ano_classification_head(ano_classes=ano_classes)  # 分类头 / Classification head
        self.reg_head = ano_regression_head(clip_number=clip_number)  # 回归头 / Regression head

    def forward(self, x):
        """
        Forward pass of AED_T model / AED_T模型前向传播
        Args:
            x: 输入特征序列 / Input feature sequence
        Returns:
            cls_scores: 分类得分 / Classification scores
            reg_scores: 回归得分 / Regression scores
        """
        # 如果输入是3D张量，重塑为5D / If input is 3D tensor, reshape to 5D
        if len(x.shape) == 3:
            x = x.view(x.shape[0], x.shape[1], -1, 4, 4)
        
        T = x.shape[0]  # 时间步数 / Number of time steps
        batch_size = x.shape[1]  # 批大小 / Batch size
        spatial_size = x.size()[3:]  # 空间尺寸 / Spatial dimensions
        
        # 初始化多层状态 / Initialize multi-layer state
        state_size = [self.num_layers] + [batch_size, self.hidden_size] + list(spatial_size)
        prev_state = [
            Variable(x.new_zeros(state_size)),  # 隐藏状态 / Hidden state
            Variable(x.new_zeros(state_size))   # 细胞状态 / Cell state
        ]
        
        # 时间序列处理 - 双层ConvLSTM / Time series processing - Two-layer ConvLSTM
        for t in range(0, T):
            h1_t, c1_t = self.convlstm1(x[t], prev_state[0])  # 第一层 / First layer
            h2_t, c2_t = self.convlstm2(h1_t, prev_state[1])  # 第二层 / Second layer
            prev_state = [torch.stack([h1_t, c1_t]), torch.stack([h2_t, c2_t])]  # 更新状态 / Update state
        
        # 特征处理和预测 / Feature processing and prediction
        x = prev_state[1][0].view(batch_size, -1)  # 展平第二层输出 / Flatten second layer output
        x = self.act(self.fc_down(x))  # 降维 + 激活 / Dimension reduction + activation
        cls_scores = self.cls_head(x)  # 分类预测 / Classification prediction
        reg_scores = self.reg_head(x)  # 回归预测 / Regression prediction
        
        return cls_scores, reg_scores



def weights_init(m):
    """
    Initialize model weights / 初始化模型权重
    Uses Xavier uniform initialization for Conv and Linear layers / 对卷积和全连接层使用Xavier均匀初始化
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)  # Xavier均匀初始化 / Xavier uniform initialization
        m.bias.data.fill_(0)  # 偏置初始化为0 / Initialize bias to 0


def model_generater(model_name, feature_size, arg=None):
    """
    Model generator function / 模型生成器函数
    Creates model instances based on model name / 根据模型名称创建模型实例
    Args:
        model_name: 模型名称 ('AED' or 'AED_T') / Model name
        feature_size: 特征维度 / Feature dimension
        arg: 参数对象 / Argument object
    Returns:
        model: 创建的模型实例 / Created model instance
    """
    if model_name == 'AED':
        model = AED(clip_number=arg.max_seqlen, input_size=feature_size//16, hidden_size=128, ano_classes=arg.ano_class)
    elif model_name == 'AED_T':
        model = AED_T(clip_number=arg.max_seqlen, input_size=feature_size//16, hidden_size=128, ano_classes=arg.ano_class)
    else:
        raise ('model_name is out of option')  # 模型名称不在选项中 / Model name not in options
    return model
