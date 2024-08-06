import torch
import torch.nn.functional as F
from ptdas.utils import plot_das_as_heatmap
from ptdas.data import DASDataset
import os

def test(model, infer_dir, device):
    model.eval()
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            for filename in os.listdir(infer_dir)[:2]:
                file_path = os.path.join(infer_dir, filename)
                
                data = DASDataset.load_das_file_no_time(file_path)
                data = DASDataset._apply_normalization(data)
                
                d = torch.from_numpy(data).to(torch.float16).unsqueeze(0).to(device)
                
                out = model(d)[0]
                
                plot_das_as_heatmap(
                    out.cpu().numpy().squeeze(),
                    filename,
                    show=False,
                    path=f"/cluster/home/jorgenaf/TinyDAS/figs/{model.__class__.__name__.lower()}/after/{filename[:-5]}.png"
                )
                
                mse = F.mse_loss(d, out).item()
                print(f"File: {filename}, MSE: {mse}")
    
    print("Processing complete.")