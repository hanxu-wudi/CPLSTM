
import function_tower 
import model_tower
from random import SystemRandom
import os 
def Model_selection_part(args,input_channels,output_channels,device):
    model_name = args.model
    hidden_channels = args.h_channels
    experiment_id = int(SystemRandom().random()*100000)
    file_path = os.path.dirname(os.path.abspath(__file__)) +"/"+str(args.dataset)

    func = function_tower.LSTM_CDE_Delay(input_channels, hidden_channels, model_name, file_path=file_path, rnd=experiment_id, time_max=args.seq_len)
    model = model_tower.CPLSTM(func=func, args=args, input_channels=input_channels, hidden_channels=hidden_channels, output_channels=output_channels, pred_len=args.pred_len,device=device,
    file_path=file_path, rnd=experiment_id, alpha=args.alpha, beta=args.beta, interpolation=args.interpolation)
    return model
    