import torch
assert torch.cuda.is_available()

def get_initial_value(model, data):
    device = data.device
    model.to(device)
    data = data.to(device)

    with torch.no_grad():
        model.eval()
        x_bar, hidden = model.ae(data)
        torch.cuda.empty_cache()

    return x_bar.cpu(), hidden.cpu()

