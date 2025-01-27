import torch
import numpy as np

global dhdt_list
global t_list
dhdt_list = torch.Tensor()
t_list = []
import os


class ContPeepholeLSTMFunc_Delay(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, file_path, rnd, time_max):
        super(ContPeepholeLSTMFunc_Delay, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # 定义 LSTM 的权重矩阵
        self.W_i = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_f = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_o = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_c = torch.nn.Linear(input_channels, hidden_channels, bias=False)

        self.U_i = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_f = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_o = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_c = torch.nn.Linear(hidden_channels, hidden_channels)

        self.P_i = torch.nn.Linear(hidden_channels, hidden_channels)
        self.P_f = torch.nn.Linear(hidden_channels, hidden_channels)
        self.P_o = torch.nn.Linear(hidden_channels, hidden_channels)

        self.file = file_path
        self.rnd = rnd
        self.dhdt_list = []  # 修改为列表，存储导数数据
        self.t_list = []
        self.time_max = time_max

        # 创建必要的目录
        os.makedirs(os.path.join(self.file, 'h_past'), exist_ok=True)
        os.makedirs(os.path.join(self.file, 'c_past'), exist_ok=True)
        os.makedirs(os.path.join(self.file, 'dhpastdt'), exist_ok=True)
        os.makedirs(os.path.join(self.file, 'dcpastdt'), exist_ok=True)

        # 将计算结果存储到内存的字典
        self.memory_storage = {
            'h_past': [],
            'c_past': [],
            'dhpastdt': [],
            'dcpastdt': []
        }

    def forward(self, t, x, h, c, dxdt):
        self.time = t.item()

        # 获取当前设备
        device = h.device

        if t == 0:
            h_past = h

            # 检查 c 的维度
            if c.shape[-1] == 0:
                # c 的第二个维度为 0，需要初始化
                batch_size = h.size(0)
                c_past = torch.randn(batch_size, self.hidden_channels, device=device, dtype=h.dtype)
                c_past = c_past.clamp(-1, 1)
            else:
                c_past = c

            dhpast_dt = torch.zeros_like(h)
            dcpast_dt = torch.zeros_like(c_past)
        else:
            # 从内存中加载过去的值，并确保数据在正确的设备上
            h_past = self.memory_storage['h_past'][-1].to(device) if self.memory_storage['h_past'] else h
            c_past = self.memory_storage['c_past'][-1].to(device) if self.memory_storage['c_past'] else c
            dhpast_dt = self.memory_storage['dhpastdt'][-1].to(device) if self.memory_storage['dhpastdt'] else torch.zeros_like(h)
            dcpast_dt = self.memory_storage['dcpastdt'][-1].to(device) if self.memory_storage['dcpastdt'] else torch.zeros_like(c)
            del self.memory_storage['h_past'][-1]
            del self.memory_storage['c_past'][-1]
            del self.memory_storage['dhpastdt'][-1]
            del self.memory_storage['dcpastdt'][-1]

        # 计算门控和候选记忆
        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_past) + self.P_i(c_past))
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_past) + self.P_f(c_past))
        c_tilde_t = torch.tanh(self.W_c(x) + self.U_c(h_past))
        # 更新记忆细胞状态
        c_t = f_t * c_past + i_t * c_tilde_t
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_past) + self.P_o(c_t))
        h_t = o_t * torch.tanh(c_t)

        # 将当前的 h_t 和 c_t 存储到内存中
        self.memory_storage['h_past'].append(h_t.cpu().detach())
        self.memory_storage['c_past'].append(c_t.cpu().detach())

        # 计算导数
        control_gradient = dxdt.derivative(t)

        # 计算各个激活项的导数
        dAdt = ((self.W_i.weight @ control_gradient.unsqueeze(-1)) + (
                    self.U_i.weight @ dhpast_dt.unsqueeze(-1)) + (
                    self.P_i.weight @ dhpast_dt.unsqueeze(-1))).squeeze(-1)
        didt = torch.mul(torch.mul(i_t, (1 - i_t)), dAdt)

        dBdt = ((self.W_f.weight @ control_gradient.unsqueeze(-1)) + (
                    self.U_f.weight @ dhpast_dt.unsqueeze(-1)) +(
                    self.P_f.weight @ dhpast_dt.unsqueeze(-1))).squeeze(-1)
        dfdt = torch.mul(torch.mul(f_t, (1 - f_t)), dBdt)

        dDdt = ((self.W_c.weight @ control_gradient.unsqueeze(-1)) + (
                    self.U_c.weight @ dhpast_dt.unsqueeze(-1))).squeeze(-1)
        dc_tilde_dt = (1 - c_tilde_t ** 2) * dDdt

        # 计算记忆细胞状态的导数
        dc_dt = dfdt * c_past + f_t * dcpast_dt + didt * c_tilde_t + i_t * dc_tilde_dt

        dCdt = ((self.W_o.weight @ control_gradient.unsqueeze(-1)) + (
                    self.U_o.weight @ dhpast_dt.unsqueeze(-1)) + (
                    self.P_o.weight @ dc_dt.unsqueeze(-1))).squeeze(-1)
        dodt = torch.mul(torch.mul(o_t, (1 - o_t)), dCdt)
        # 计算隐藏状态的导数
        d_tanh_c_t = (1 - torch.tanh(c_t) ** 2) * dc_dt
        dh_dt = dodt * torch.tanh(c_t) + o_t * d_tanh_c_t

        # 将导数存储到内存中
        self.memory_storage['dhpastdt'].append(dh_dt.cpu().detach())
        self.memory_storage['dcpastdt'].append(dc_dt.cpu().detach())

        if self.time % 1 == 0 and self.time not in self.t_list:
            self.t_list.append(self.time)
            if len(self.dhdt_list) > 0:
                self.dhdt_list = torch.cat([self.dhdt_list, dh_dt.unsqueeze(0)], dim=0)
            else:
                self.dhdt_list = dh_dt.unsqueeze(0)

            if self.time_max - self.time <= 1:
                # 将数据从内存保存到文件
                np.save(self.file + '/dhpastdt/dhdt_' + str(self.rnd) + '.npy', self.dhdt_list.cpu().detach().numpy())
        else:
            if (self.time_max > self.time_max - 1 and self.time == 0) or (self.time_max < self.time_max - 1 and self.time == 1):
                self.t_list = []
                self.dhdt_list = dh_dt.unsqueeze(0)
                self.dhdt_list = torch.Tensor()

        # 更新 dhdt_list 和 t_list
        return dh_dt

class ContLSTMFunc_Delay(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, file_path, rnd, time_max):
        super(ContLSTMFunc_Delay, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # 定义 LSTM 的权重矩阵
        self.W_i = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_f = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_o = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_c = torch.nn.Linear(input_channels, hidden_channels, bias=False)

        self.U_i = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_f = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_o = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_c = torch.nn.Linear(hidden_channels, hidden_channels)

        self.file = file_path
        self.rnd = rnd
        self.dhdt_list = []  # 修改为列表，存储导数数据
        self.t_list = []
        self.time_max = time_max

        # 创建必要的目录
        os.makedirs(os.path.join(self.file, 'h_past'), exist_ok=True)
        os.makedirs(os.path.join(self.file, 'c_past'), exist_ok=True)
        os.makedirs(os.path.join(self.file, 'dhpastdt'), exist_ok=True)
        os.makedirs(os.path.join(self.file, 'dcpastdt'), exist_ok=True)

        # 将计算结果存储到内存的字典
        self.memory_storage = {
            'h_past': [],
            'c_past': [],
            'dhpastdt': [],
            'dcpastdt': []
        }

    def forward(self, t, x, h, c, dxdt):
        self.time = t.item()

        # 获取当前设备
        device = h.device

        if t == 0:
            h_past = h

            # 检查 c 的维度
            if c.shape[-1] == 0:
                # c 的第二个维度为 0，需要初始化
                batch_size = h.size(0)
                c_past = torch.randn(batch_size, self.hidden_channels, device=device, dtype=h.dtype)
                c_past = c_past.clamp(-1, 1)
            else:
                c_past = c

            dhpast_dt = torch.zeros_like(h)
            dcpast_dt = torch.zeros_like(c_past)
        else:
            # 从内存中加载过去的值，并确保数据在正确的设备上
            h_past = self.memory_storage['h_past'][-1].to(device) if self.memory_storage['h_past'] else h
            c_past = self.memory_storage['c_past'][-1].to(device) if self.memory_storage['c_past'] else c
            dhpast_dt = self.memory_storage['dhpastdt'][-1].to(device) if self.memory_storage['dhpastdt'] else torch.zeros_like(h)
            dcpast_dt = self.memory_storage['dcpastdt'][-1].to(device) if self.memory_storage['dcpastdt'] else torch.zeros_like(c)
            del self.memory_storage['h_past'][-1]
            del self.memory_storage['c_past'][-1]
            del self.memory_storage['dhpastdt'][-1]
            del self.memory_storage['dcpastdt'][-1]

        # 计算门控和候选记忆
        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_past))
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_past))
        c_tilde_t = torch.tanh(self.W_c(x) + self.U_c(h_past))
        # 更新记忆细胞状态
        c_t = f_t * c_past + i_t * c_tilde_t
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_past))
        h_t = o_t * torch.tanh(c_t)

        # 将当前的 h_t 和 c_t 存储到内存中
        self.memory_storage['h_past'].append(h_t.cpu().detach())
        self.memory_storage['c_past'].append(c_t.cpu().detach())

        # 计算导数
        control_gradient = dxdt.derivative(t)

        # 计算各个激活项的导数
        dAdt = ((self.W_i.weight @ control_gradient.unsqueeze(-1)) + (
                    self.U_i.weight @ dhpast_dt.unsqueeze(-1))).squeeze(-1)
        didt = torch.mul(torch.mul(i_t, (1 - i_t)), dAdt)

        dBdt = ((self.W_f.weight @ control_gradient.unsqueeze(-1)) + (
                    self.U_f.weight @ dhpast_dt.unsqueeze(-1))).squeeze(-1)
        dfdt = torch.mul(torch.mul(f_t, (1 - f_t)), dBdt)

        dDdt = ((self.W_c.weight @ control_gradient.unsqueeze(-1)) + (
                    self.U_c.weight @ dhpast_dt.unsqueeze(-1))).squeeze(-1)
        dc_tilde_dt = (1 - c_tilde_t ** 2) * dDdt

        # 计算记忆细胞状态的导数
        dc_dt = dfdt * c_past + f_t * dcpast_dt + didt * c_tilde_t + i_t * dc_tilde_dt

        dCdt = ((self.W_o.weight @ control_gradient.unsqueeze(-1)) + (
                    self.U_o.weight @ dhpast_dt.unsqueeze(-1))).squeeze(-1)
        dodt = torch.mul(torch.mul(o_t, (1 - o_t)), dCdt)
        # 计算隐藏状态的导数
        d_tanh_c_t = (1 - torch.tanh(c_t) ** 2) * dc_dt
        dh_dt = dodt * torch.tanh(c_t) + o_t * d_tanh_c_t

        # 将导数存储到内存中
        self.memory_storage['dhpastdt'].append(dh_dt.cpu().detach())
        self.memory_storage['dcpastdt'].append(dc_dt.cpu().detach())

        if self.time % 1 == 0 and self.time not in self.t_list:
            self.t_list.append(self.time)
            if len(self.dhdt_list) > 0:
                self.dhdt_list = torch.cat([self.dhdt_list, dh_dt.unsqueeze(0)], dim=0)
            else:
                self.dhdt_list = dh_dt.unsqueeze(0)

            if self.time_max - self.time <= 1:
                # 将数据从内存保存到文件
                np.save(self.file + '/dhpastdt/dhdt_' + str(self.rnd) + '.npy', self.dhdt_list.cpu().detach().numpy())
        else:
            if (self.time_max > self.time_max - 1 and self.time == 0) or (self.time_max < self.time_max - 1 and self.time == 1):
                self.t_list = []
                self.dhdt_list = dh_dt.unsqueeze(0)
                self.dhdt_list = torch.Tensor()

        # 更新 dhdt_list 和 t_list
        return dh_dt


class ContGruFunc_Delay(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, file_path, rnd, time_max):
        super(ContGruFunc_Delay, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.W_r = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_z = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_h = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.U_r = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_z = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_h = torch.nn.Linear(hidden_channels, hidden_channels)
        self.file = file_path
        self.rnd = rnd
        self.dhdt_list = dhdt_list
        self.t_list = t_list
        self.time_max = time_max

        os.makedirs(os.path.join(self.file, 'h_past'), exist_ok=True)
        os.makedirs(os.path.join(self.file, 'c_past'), exist_ok=True)
        os.makedirs(os.path.join(self.file, 'dhpastdt'), exist_ok=True)
        os.makedirs(os.path.join(self.file, 'dcpastdt'), exist_ok=True)

    def forward(self, t, x, h, dxdt):

        self.time = t.item()
        if t == 0:
            h_past = h
        else:
            h_past = torch.Tensor(np.load(self.file + "/h_past/h_past_" + str(self.rnd) + ".npy")).to(h)

        r = self.W_r(x) + self.U_r(h_past)

        r = r.sigmoid()
        z = self.W_z(x) + self.U_z(h_past)
        z = z.sigmoid()
        g0 = self.W_h(x) + self.U_h(r * h_past)
        g = g0.tanh()
        h_ = torch.mul(z, h_past) + torch.mul((1 - z), g)  # save h at t
        np.save(self.file + '/h_past/h_past_' + str(self.rnd) + '.npy', h_.cpu().detach().numpy())

        hg = h_past - g

        if t == 0:
            dhpast_dt = (1 - z) * (g - h)
        else:

            dhpast_dt = torch.Tensor(np.load(self.file + "/dhpastdt/dhpastdt_" + str(self.rnd) + ".npy")).to(h)

        control_gradient = dxdt.derivative(t)  # 256,28
        dAdt = ((self.W_z.weight @ control_gradient.unsqueeze(-1)) + (
                    self.U_z.weight @ dhpast_dt.unsqueeze(-1))).squeeze(-1)  # dAdt = 10,49,1
        dzdt = torch.mul(torch.mul(z, (1 - z)), dAdt)
        drdt = torch.mul(torch.mul(r, (1 - r)), ((self.W_r.weight @ control_gradient.unsqueeze(-1)) + (
                    self.U_r.weight @ dhpast_dt.unsqueeze(-1))).squeeze(-1))  # drdt : 10,49
        dBdt = (self.W_h.weight @ control_gradient.unsqueeze(-1)).squeeze(-1) + torch.mul(
            (self.U_h.weight @ drdt.unsqueeze(-1)).squeeze(-1), h) + torch.mul(
            (self.U_h.weight @ r.unsqueeze(-1)).squeeze(-1), dhpast_dt)
        dgdt = torch.mul(torch.mul((1 - g), (1 + g)), dBdt)
        dhgdt = dhpast_dt - dgdt

        dhdt = torch.mul(dzdt, hg) + torch.mul(z, dhgdt) + dgdt
        np.save(self.file + '/dhpastdt/dhpastdt_' + str(self.rnd) + '.npy', dhdt.cpu().detach().numpy())

        if self.time % 1 == 0 and self.time not in self.t_list:

            self.t_list.append(self.time)
            if self.dhdt_list.shape[0] > 0:

                self.dhdt_list = torch.cat([self.dhdt_list, dhdt.unsqueeze(0)], dim=0)

            else:

                self.dhdt_list = dhdt.unsqueeze(0)
            if self.time_max - self.time <= 1:
                np.save(self.file + '/dhpastdt/dhdt_' + str(self.rnd) + '.npy', self.dhdt_list.cpu().detach().numpy())
                # self.dhdt_list = torch.Tensor()
                # self.t_list = []
        else:

            if (self.time_max > self.time_max - 1 and self.time == 0) or (
                    self.time_max < self.time_max - 1 and self.time == 1):
                self.t_list = []

                self.dhdt_list = dhdt.unsqueeze(0)
                self.dhdt_list = torch.Tensor()

        return dhdt

class ContinuousDelayRNNConverter(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, model):
        super(ContinuousDelayRNNConverter, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model
        self.linear = torch.nn.Linear(self.hidden_channels, self.input_channels + self.hidden_channels)

    def forward(self, t, z, dxdt):
        x = z[..., : self.input_channels]

        h = z[..., self.input_channels: self.input_channels + self.hidden_channels]

        c = z[..., self.input_channels + self.hidden_channels: self.input_channels + self.hidden_channels * 2]

        # 确保 h 和 c 的维度正确
        h = h.clamp(-1, 1)


        # 调用模型，计算 dh/dt 和 dc/dt
        model_out = self.model(t, x, h, c, dxdt)

        out = self.linear(model_out)

        # 返回 dz/dt
        return model_out, out
# class ContinuousDelayRNNConverter(torch.nn.Module):
#     def __init__(self, input_channels, hidden_channels, model):
#         super(ContinuousDelayRNNConverter, self).__init__()
#
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.model = model
#         self.linear = torch.nn.Linear(self.hidden_channels, self.input_channels + self.hidden_channels)
#
#     def forward(self, t, z, dxdt):
#         x = z[..., : self.input_channels]
#         h = z[..., self.input_channels:]
#         h = h.clamp(-1, 1)
#
#         model_out = self.model(t, x, h, dxdt)  # 1024,49
#         out = self.linear(model_out)
#
#         return model_out, out

def LSTM_CDE_Delay(input_channels, hidden_channels, model_name, file_path, rnd, time_max=None):
    if model_name == 'LSTM':
       func = ContLSTMFunc_Delay(input_channels=input_channels, hidden_channels=hidden_channels, file_path=file_path,
                              rnd=rnd, time_max=time_max)

    elif model_name == 'contime':
       func = ContGruFunc_Delay(input_channels=input_channels, hidden_channels=hidden_channels, file_path=file_path,
                              rnd=rnd, time_max=time_max)

    elif model_name == 'peepholeLSTM':
        func = ContPeepholeLSTMFunc_Delay(input_channels=input_channels, hidden_channels=hidden_channels,
                                          file_path=file_path,
                                          rnd=rnd, time_max=time_max)

    return ContinuousDelayRNNConverter(input_channels=input_channels,
                                       hidden_channels=hidden_channels,
                                       model=func)


